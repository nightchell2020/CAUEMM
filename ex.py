import re
import os



# # 예시 파일 목록 (실제로는 os.listdir() 등을 통해 가져올 수 있음)
# file_names = os.listdir("/media/night/dawn/brain/MRI/filtered_files/filtered")
# # 숫자 부분 추출
# numbers = []
# for name in file_names:
#     match = re.search(r'(\d{6})', name)  # 6자리 숫자 추출
#     if match:
#         numbers.append(int(match.group(1)))
#
# # 중복 제거 및 정렬
# numbers = sorted(set(numbers))
#
# # 누락된 숫자 찾기
# missing = [i for i in range(numbers[0], 1389) if i not in numbers]
#
# # 출력
# print("누락된 번호:")
# print(missing)
# print(len(missing))

# #### gz 파일 압축해 ###
# import nibabel as nib
#
# def decompress_and_delete_nii_gz(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith(".nii.gz"):
#             gz_path = os.path.join(directory, filename)
#             nii_filename = filename.replace(".nii.gz", ".nii")
#             nii_path = os.path.join(directory, nii_filename)
#
#             print(f"📂 압축 해제 중: {filename} → {nii_filename}")
#             try:
#                 # 압축 해제 및 저장
#                 img = nib.load(gz_path)
#                 nib.save(img, nii_path)
#
#                 # 원본 삭제
#                 os.remove(gz_path)
#                 print(f"🗑️ 원본 삭제 완료: {filename}")
#             except Exception as e:
#                 print(f"⚠️ 오류 발생: {filename} → {e}")
#
#     print("\n✅ 모든 .nii.gz 파일 압축 해제 및 삭제 완료!")
#
# # 경로 설정
# target_dir = "/media/night/dawn/brain/MRI/filtered_files"
# decompress_and_delete_nii_gz(target_dir)
# #########################


# import json
# from datasets.cauemm_dataset import CauEegMriMultiModalDataset
#
# # with open(
# #         os.path.join('/media/night/dawn/brain/CAUEEG/caueeg-dataset/', "annotation.json"),
# #         "r",
# # ) as json_file:
# #     annotation = json.load(json_file)
# with open(
#         os.path.join('/media/night/dawn/brain/CAUEEG/caueeg-dataset/', 'abnormal' + ".json"),
#         "r",
# ) as json_file:
#     task_dict = json.load(json_file)
# t_d = CauEegMriMultiModalDataset(
#     '/media/night/dawn/brain/,',
#     task_dict["train_split"],
#     load_event = False,
#     eeg_file_format='memmap',
#     transform=None
# )
# print("@@@@")
