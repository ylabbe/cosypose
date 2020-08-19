import pandas as pd
import torch
from collections import defaultdict
import cosypose.utils.tensor_collection as tc
from cosypose.lib3d.transform_ops import invert_T


def parse_obs_data(obs):
    data = defaultdict(list)
    frame_info = obs['frame_info']
    TWC = torch.as_tensor(obs['camera']['TWC']).float()
    for n, obj in enumerate(obs['objects']):
        info = dict(frame_obj_id=n,
                    label=obj['name'],
                    visib_fract=obj.get('visib_fract', 1),
                    scene_id=frame_info['scene_id'],
                    view_id=frame_info['view_id'])
        data['infos'].append(info)
        data['TWO'].append(obj['TWO'])
        data['bboxes'].append(obj['bbox'])

    for k, v in data.items():
        if k != 'infos':
            data[k] = torch.stack([torch.as_tensor(x) .float()for x in v])

    data['infos'] = pd.DataFrame(data['infos'])
    TCO = invert_T(TWC).unsqueeze(0) @ data['TWO']

    data = tc.PandasTensorCollection(
        infos=data['infos'],
        TCO=TCO,
        bboxes=data['bboxes'],
        poses=TCO,
    )
    return data


def data_to_pose_model_inputs(data):
    TXO = data.poses
    obj_infos = []
    for n in range(len(data)):
        obj_info = dict(name=data.infos.loc[n, 'label'])
        obj_infos.append(obj_info)
    return TXO, obj_infos
