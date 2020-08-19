#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <random>
#include <tuple>
#include <unordered_map>


namespace py = pybind11;

namespace cosypose {
struct Match {
  int c1, c2;
};

using ViewPair = std::tuple<int, int>;
using Matches = std::vector<Match>;

std::vector<int> sort_indexes(const std::vector<float> &v) {
  std::vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
  return idx;
}

std::vector<int> random_permutation(int N, int seed) {
  std::vector<int> vec;
  for (int i = 0; i < N; i++) {
    vec.push_back(i);
  }
  std::shuffle(vec.begin(), vec.end(), std::default_random_engine(seed));
  return vec;
}

py::tuple make_ransac_infos(
    std::vector<int>& view_ids,
    std::vector<std::string>& labels,
    int n_ransac_iter = 100,
    int seed = 0) {
  // Make tentative matches.
  std::map<ViewPair, Matches> tentative_matches_per_view_pair;
  int n_cand = view_ids.size();
  for (int n = 0; n < n_cand; n++) {
    for (int m = 0; m < n_cand; m++) {
      if (view_ids[n] != view_ids[m] && labels[n] == labels[m]) {
        ViewPair view_pair(view_ids[n], view_ids[m]);
        Match tentative_match({n, m});
        tentative_matches_per_view_pair[view_pair].push_back(tentative_match);
      }
    }
  }

  // Ransac seeds
  std::vector<int> seed_view1, seed_view2;
  std::vector<int> seed_match1_cand1, seed_match1_cand2, seed_match2_cand1,
      seed_match2_cand2;
  std::vector<int> mtc_hypothesis_id, mtc_cand1, mtc_cand2;

  int n_ransac_seeds;
  n_ransac_seeds = 0;
  for (auto kv : tentative_matches_per_view_pair) {
    const Matches& tentative_matches = kv.second;
    int n_tentative_matches = tentative_matches.size();
    auto perm1 = random_permutation(n_tentative_matches, seed);
    auto perm2 = random_permutation(n_tentative_matches, seed + 1);
    int n_pairs = 0;
    // Ransac seeds
    for (int m1_id : perm1) {
      if (n_pairs >= n_ransac_iter)
        break;
      for (int m2_id : perm2) {
        if (n_pairs >= n_ransac_iter)
          break;
        if (m1_id != m2_id) {
          seed_view1.push_back(std::get<0>(kv.first));
          seed_view2.push_back(std::get<1>(kv.first));
          seed_match1_cand1.push_back(tentative_matches[m1_id].c1);
          seed_match1_cand2.push_back(tentative_matches[m1_id].c2);
          seed_match2_cand1.push_back(tentative_matches[m2_id].c1);
          seed_match2_cand2.push_back(tentative_matches[m2_id].c2);
          for (int i = 0; i < n_tentative_matches; i++) {
            mtc_hypothesis_id.push_back(n_ransac_seeds);
            mtc_cand1.push_back(tentative_matches[i].c1);
            mtc_cand2.push_back(tentative_matches[i].c2);
          }
          n_pairs++;
          n_ransac_seeds++;
        }
      }
    }
  }
  py::dict matches, seeds, mtc;
  seeds["view1"] = py::array_t<int>(py::cast(seed_view1));
  seeds["view2"] = py::array_t<int>(py::cast(seed_view2));
  seeds["match1_cand1"] = py::array_t<int>(py::cast(seed_match1_cand1));
  seeds["match1_cand2"] = py::array_t<int>(py::cast(seed_match1_cand2));
  seeds["match2_cand1"] = py::array_t<int>(py::cast(seed_match2_cand1));
  seeds["match2_cand2"] = py::array_t<int>(py::cast(seed_match2_cand2));
  mtc["hypothesis_id"] = py::array_t<int>(py::cast(mtc_hypothesis_id));
  mtc["cand1"] = py::array_t<int>(py::cast(mtc_cand1));
  mtc["cand2"] = py::array_t<int>(py::cast(mtc_cand2));
  py::tuple outputs = py::make_tuple(seeds, mtc);
  return outputs;
}

py::dict find_ransac_inliers(
    const py::array_t<int> seeds_view1,
    const py::array_t<int> seeds_view2,
    const py::array_t<int> mtc_hypothesis_id,
    const py::array_t<int> mtc_cand1,
    const py::array_t<int> mtc_cand2,
    const py::array_t<float> dists,
    float dist_threshold,
    int n_min_inliers) {

  struct RansacHypothesis {
    int hypothesis_id;
    int view1, view2;
    Matches matches_inliers;
    Matches matches_inliers_uniqs;
    std::vector<float> matches_inliers_dists;
    float dists_sum;
    int n_inliers;
  };

  auto seeds_view1_ = seeds_view1.unchecked<1>();
  auto seeds_view2_ = seeds_view2.unchecked<1>();
  auto mtc_hypothesis_id_= mtc_hypothesis_id.unchecked<1>();
  auto mtc_cand1_ = mtc_cand1.unchecked<1>();
  auto mtc_cand2_ = mtc_cand2.unchecked<1>();
  auto dists_ = dists.unchecked<1>();

  // A) Iterate over all seeds views. Build id_to_hypothesis and
  // viewpair_to_hypothesis [pointer here].
  std::unordered_map<int, RansacHypothesis> id_to_hypothesis;
  std::map<ViewPair, std::vector<int>> viewpair_to_hypotheses_ids;
  int n_hypotheses = seeds_view1.size();
  for (int n = 0; n < n_hypotheses; n++) {
    int v1 = seeds_view1_(n);
    int v2 = seeds_view2_(n);
    ViewPair view_pair(v1, v2);
    RansacHypothesis hypothesis;
    hypothesis.hypothesis_id = n;
    hypothesis.view1 = v1;
    hypothesis.view2 = v2;
    hypothesis.dists_sum = 0.f;
    hypothesis.n_inliers = 0;
    hypothesis.matches_inliers_dists.clear();
    hypothesis.matches_inliers.clear();
    hypothesis.matches_inliers_uniqs.clear();
    id_to_hypothesis[n] = hypothesis;
    viewpair_to_hypotheses_ids[view_pair].push_back(n);
  }

  // B) Iterate over mtc. Fill the hypotheses scores etc.
  int n_mtc = mtc_hypothesis_id.size();
  for (int n = 0; n < n_mtc; n++) {
    RansacHypothesis& hypothesis = id_to_hypothesis[mtc_hypothesis_id_(n)];
    const float& dist = dists_(n);
    if (dist <= dist_threshold) {
      Match match({mtc_cand1_(n), mtc_cand2_(n)});
      hypothesis.matches_inliers.push_back(match);
      hypothesis.matches_inliers_dists.push_back(dist);
    }
  }

  // C) Find the best pairs for each hypothesis.
  for (auto kv : viewpair_to_hypotheses_ids) {
    for (auto hypothesis_id : kv.second) {
      RansacHypothesis& hypothesis = id_to_hypothesis[hypothesis_id];
      std::set<int> cand1_matched, cand2_matched;
      for (auto i: sort_indexes(hypothesis.matches_inliers_dists)) {
        Match& match = hypothesis.matches_inliers[i];
        if ((cand1_matched.count(match.c1) == 0) &&
            (cand2_matched.count(match.c2) == 0)){
          cand1_matched.insert(match.c1);
          cand2_matched.insert(match.c2);
          hypothesis.matches_inliers_uniqs.push_back(match);
          hypothesis.dists_sum += hypothesis.matches_inliers_dists[i];
          hypothesis.n_inliers += 1;
        }
      }
    }
  }

  // D) Find the best hypotheses.
  std::vector<int> inlier_matches_cand1, inlier_matches_cand2, best_hypotheses;
  for (auto kv : viewpair_to_hypotheses_ids) {
    RansacHypothesis best_hypothesis;
    best_hypothesis.hypothesis_id = -1;
    best_hypothesis.dists_sum = std::numeric_limits<float>::max();
    best_hypothesis.n_inliers = 0;
    for (auto hypothesis_id : kv.second) {
      const RansacHypothesis& hypothesis = id_to_hypothesis[hypothesis_id];
      if ((hypothesis.n_inliers >= n_min_inliers) &&
          ((hypothesis.n_inliers > best_hypothesis.n_inliers) ||
           (hypothesis.n_inliers == best_hypothesis.n_inliers &&
            hypothesis.dists_sum < best_hypothesis.dists_sum))) {
        best_hypothesis = hypothesis;
      }
    }
    if (best_hypothesis.hypothesis_id > 0) {
      best_hypotheses.push_back(best_hypothesis.hypothesis_id);
      for (auto match : best_hypothesis.matches_inliers_uniqs) {
        inlier_matches_cand1.push_back(match.c1);
        inlier_matches_cand2.push_back(match.c2);
      }
    }
  }
  py::dict outputs;
  outputs["inlier_matches_cand1"] = py::array_t<int>(py::cast(inlier_matches_cand1));
  outputs["inlier_matches_cand2"] = py::array_t<int>(py::cast(inlier_matches_cand2));
  outputs["best_hypotheses"] = py::array_t<int>(py::cast(best_hypotheses));
  return outputs;
}

  py::array_t<int> scatter_argmin(const py::array_t<float> array,
                                  const py::array_t<int> expand_ids) {
    auto array_ = array.unchecked<1>();
    auto expand_ids_ = expand_ids.unchecked<1>();

    std::unordered_map<int, float> lowest_values;
    std::unordered_map<int, int> best_ids;
    for (int n=0; n < array_.size(); n++){
      const int& expand_id = expand_ids_(n);
      const float& value = array_(n);
      const auto& exists = best_ids.find(expand_id);
      if ( exists == best_ids.end() ){
        best_ids[expand_id] = n;
        lowest_values[expand_id] = value;
      }
      if ( value < lowest_values[expand_id] ){
        best_ids[expand_id] = n;
        lowest_values[expand_id] = value;
      }
    }

    std::vector<int> vector_best_ids;
    for (auto n=0; n < best_ids.size(); n++){
      vector_best_ids.push_back(best_ids[n]);
    }
    auto array_best_ids = py::array_t<int>(py::cast(vector_best_ids));
    return array_best_ids;
  }

  py::tuple expand_ids_for_symmetry(const std::vector<std::string>& labels,
                                    std::unordered_map<std::string, int> n_symmetries) {
    std::vector<int> ids_expand, sym_ids;
    for (auto n=0; n < labels.size(); n++) {
      for (int k=0; k < n_symmetries[labels[n]]; k++) {
        ids_expand.push_back(n);
        sym_ids.push_back(k);
      }
    }
    auto array_ids_expand = py::array_t<int>(py::cast(ids_expand));
    auto array_sym_ids = py::array_t<int>(py::cast(sym_ids));
    return py::make_tuple(array_ids_expand, array_sym_ids);
  }
}

 // namespace cosypose

PYBIND11_MODULE(cosypose_cext, m) {
  m.def("make_ransac_infos", &cosypose::make_ransac_infos);
  m.def("find_ransac_inliers", &cosypose::find_ransac_inliers);
  m.def("scatter_argmin", &cosypose::scatter_argmin);
  m.def("expand_ids_for_symmetry", &cosypose::expand_ids_for_symmetry);
}
