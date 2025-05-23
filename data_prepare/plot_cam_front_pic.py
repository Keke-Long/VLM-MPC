from nuscenes.nuscenes import NuScenes
from PIL import Image
import os
import matplotlib.pyplot as plt

# # 加载nuScenes数据集
# dataroot = '/home/ubuntu/Documents/Nuscenes'
# nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)


def plot_cam_front_pic(scene_token_name, nusc):
    scene_folder = os.path.join('../scenes_data1', scene_token_name)

    # 确保场景文件夹存在
    if not os.path.exists(scene_folder):
        print(f"Scene folder {scene_folder} does not exist.")
    else:
        scene = nusc.get('scene', scene_token_name)
        sample_token = scene['first_sample_token']

        images = []
        count = 0  # 计数器以确保最多处理20个图像
        while sample_token and count < 40:
            sample = nusc.get('sample', sample_token)
            cam_token = sample['data']['CAM_FRONT']
            cam_data = nusc.get('sample_data', cam_token)
            cam_filepath = nusc.get_sample_data_path(cam_token)

            if os.path.exists(cam_filepath):
                img = Image.open(cam_filepath)
                images.append(img)
                count += 1
            sample_token = sample['next']

        # 检查是否有图像被加载
        if images:
            # 设置图像网格大小
            num_cols = 10
            num_rows = 4

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(30,10))  # 根据需要调整大小
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
            for idx, img in enumerate(images):
                row = idx // num_cols
                col = idx % num_cols
                if row < num_rows:
                    axes[row, col].imshow(img)
                    axes[row, col].axis('off')  # 隐藏坐标轴

            # 去除多余的子图位置
            for idx in range(len(images), num_rows * num_cols):
                row = idx // num_cols
                col = idx % num_cols
                axes[row, col].axis('off')

            # 保存拼接的图像到相应的场景文件夹下
            fig.savefig(os.path.join(scene_folder, 'cam_front_data.png'))
            plt.close(fig)
        else:
            print(f"No camera data found for scene: {scene_token_name}")


# if __name__ == "__main__":
#     scene_token_name = '0ac05652a4c44374998be876ba5cd6fd'
#     plot_cam_front_pic(scene_token_name)