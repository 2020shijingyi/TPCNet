import glob
import cv2
import os

def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))
def downsample_image(image_path, file_name,save_path):
    """
    Args:
        image_path (str):
        save_dir (str):

    Returns:
        (new_w, new_h, k_w, k_h, save_path)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"can not read img path: {image_path}")

    h, w = img.shape[:2]

    def max_divisible_power2(x):
        k = 0
        while x % 2 == 0 and x > 1:
            x //= 2
            k += 1
        return k

    k_w = max_divisible_power2(w)
    k_h = max_divisible_power2(h)

    new_w = w // 2
    new_h = h // 2

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.join(save_path,file_name), exist_ok=True)
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_path,file_name,image_name)
    cv2.imwrite(save_path, resized)

if __name__ == '__main__':
    data_path = r'F:\LCDP\lcdp_dataset-001'
    save_path = r'F:\LCDP\LCDP'
    files = glob_file_list(data_path)

    for file in files:
        file_name = os.path.basename(file)
        imgs_paths = glob_file_list(file)
        for i,imgs_path in enumerate(imgs_paths):
            print(i)
            downsample_image(imgs_path,file_name,save_path)
