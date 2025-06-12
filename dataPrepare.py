import os
import shutil
import pandas as pd
import torch
import numpy as np
import nibabel as nib

nii= nib.load("/media/night/dawn/brain/MRI/filtered_files/000001_01.nii") #.get_fdata()


# data norm #
def z_score_normalize(volume):
    """Z-score 정규화: (x - mean) / std"""
    mask = volume > 0
    mean = np.mean(volume[mask])
    std = np.std(volume[mask])
    norm = np.zeros_like(volume)
    norm[mask] = (volume[mask] - mean) / std
    return norm

def min_max_normalize(volume):
    """Min-Max 정규화: 0~1 범위로 스케일링"""
    mask = volume > 0
    min_val = np.min(volume[mask])
    max_val = np.max(volume[mask])
    norm = np.zeros_like(volume)
    norm[mask] = (volume[mask] - min_val) / (max_val - min_val)
    return norm

def to_tensor(volume):
    """(D, H, W) 형태의 NumPy → PyTorch Tensor로 변환"""
    tensor = torch.tensor(volume, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)  # (1, D, H, W) → 채널 추가
    return tensor


norm_volume = z_score_normalize(nii.get_fdata())
tensor_input = to_tensor(norm_volume)
print(f"Torch Tensor shape: {tensor_input.shape}")  # (1, D, H, W)
print("#######  I am Here   ######")
# # 1️⃣ 엑셀 파일에서 B열 숫자 리스트 가져오기
# excel_path = "/home/night/Mycode/annotation_debug.xlsx"
# df = pd.read_excel(excel_path, engine='openpyxl')
# serial_list = df.iloc[:, 1].dropna().astype(int).tolist()  # B열, NaN 제외, int형으로 변환
#
# print("추출된 serial list:", serial_list)
#
# # 2️⃣ 파일 경로 및 새로운 폴더 준비
# source_dir = "/media/night/dawn/brain/MRI/230413_MRI_Integration/"
# dest_dir = "/media/night/dawn/brain/MRI/filtered_files/filtered"  # 원하는 경로로 변경 가능
#
# if not os.path.exists(dest_dir):
#     os.makedirs(dest_dir)
#
# # 3️⃣ 파일 검색 및 복사
# for idx, serial in enumerate(serial_list):
#     serial_str = f"{serial:07d}"  # 예: 001809
#     new_prefix = f"{idx+1:05d}"   # 인덱스 기반 새 이름, 1부터 시작 (예: 00001)
#
#     for root, _, files in os.walk(source_dir):
#         for file in files:
#             if serial_str in file:
#                 # serial_str 제거 (처음으로만)
#                 new_name = file.replace(serial_str, new_prefix, 1)
#                 source_file = os.path.join(root, file)
#                 dest_file = os.path.join(dest_dir, new_name)
#
#                 shutil.copy2(source_file, dest_file)
#                 print(f"복사 및 이름 변경: {file} → {new_name}")
#
#
# print("작업 완료!")