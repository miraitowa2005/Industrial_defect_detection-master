import requests
import os
import zipfile
import shutil
from tqdm import tqdm
import git

# NEU数据集下载配置
NEU_OFFICIAL_URL = "http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html"
NEU_DOWNLOAD_URL = "http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.zip"
GITHUB_REPO_URL = "https://github.com/abin24/NEU-CLS-Surface-Defect-Database.git"
KAGGLE_URL = "https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database/download?datasetVersionNumber=1"
DATASET_DIR = "d:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset"
TEMP_DIR = "temp_neu_dataset"

# NEU数据集类别映射
NEU_CLASSES = {
    "RS": "0_scratch",  # 轧制氧化皮 -> 划痕类
    "Pa": "1_broken",   # 斑块 -> 破损类
    "Cr": "2_crack",    # 开裂 -> 裂缝类
    "PS": "3_dent",     # 点蚀表面 -> 凹痕类
    "In": "4_discolor", # 内含物 -> 变色类
    "Sc": "5_pinhole"   # 划痕 -> 针孔类
}

def download_file(url, save_path):
    """从指定URL下载文件"""
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def download_from_github(repo_url, save_path):
    """从GitHub仓库克隆数据集"""
    print("从GitHub仓库克隆数据集...")
    try:
        git.Repo.clone_from(repo_url, save_path)
        return True
    except Exception as e:
        print(f"GitHub克隆失败: {e}")
        return False

def download_from_kaggle(url, save_path):
    """从Kaggle下载数据集"""
    print("Kaggle下载需要登录账号，正在尝试直接下载...")
    try:
        # 尝试使用requests下载
        return download_file(url, save_path)
    except Exception as e:
        print(f"直接下载失败: {e}")
        return False

def extract_zip(zip_path, extract_path):
    """解压zip文件"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def organize_dataset(source_dir, target_dir, class_mapping):
    """将下载的数据集组织成项目需要的格式"""
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 获取所有图像文件
    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))
    
    # 按类别复制图像
    print(f"开始组织数据集，共 {len(image_files)} 张图像")
    
    for image_path in tqdm(image_files):
        filename = os.path.basename(image_path)
        
        # 提取类别标签 (RS, Pa, Cr, PS, In, Sc)
        class_label = filename.split('_')[0]
        
        if class_label in class_mapping:
            # 获取目标类别目录
            target_class_dir = os.path.join(target_dir, class_mapping[class_label])
            
            # 确保目标类别目录存在
            if not os.path.exists(target_class_dir):
                os.makedirs(target_class_dir)
            
            # 复制图像到目标目录
            target_path = os.path.join(target_class_dir, filename)
            shutil.copy2(image_path, target_path)
        else:
            print(f"未知类别: {class_label}，文件: {filename}")
    
    print("数据集组织完成！")

def create_dummy_data(target_dir, num_images_per_class=100):
    """创建虚拟数据集作为备选方案"""
    print(f"\n正在创建虚拟数据集，每个类别 {num_images_per_class} 张图像...")
    
    import numpy as np
    from PIL import Image
    
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 虚拟类别
    dummy_classes = [
        "0_scratch",
        "1_broken",
        "2_crack",
        "3_dent",
        "4_discolor",
        "5_pinhole",
        "6_burr",
        "7_foreign",
        "8_pin"
    ]
    
    for class_name in tqdm(dummy_classes):
        class_dir = os.path.join(target_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # 创建虚拟图像
        for i in range(num_images_per_class):
            # 创建随机图像 (224x224)
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # 添加一些模拟缺陷
            if "scratch" in class_name:
                # 添加划痕
                img_array[100:150, 50:150, :] = 0
            elif "crack" in class_name:
                # 添加裂缝
                img_array[50:150, 100, :] = 0
            elif "dent" in class_name:
                # 添加凹痕
                for x in range(80, 120):
                    for y in range(80, 120):
                        if (x-100)**2 + (y-100)**2 < 100:
                            img_array[x, y, :] = 100
            
            # 保存图像
            img = Image.fromarray(img_array)
            img.save(os.path.join(class_dir, f"{class_name}_{i}.jpg"))
    
    print("虚拟数据集创建完成！")
    return True

def main():
    print("=== NEU表面缺陷数据集下载工具 ===")
    print(f"目标目录: {DATASET_DIR}")
    
    # 确保临时目录存在
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    download_success = False
    
    # 尝试从NEU官方网站下载
    print("\n1. 尝试从NEU官方网站下载真实NEU数据集...")
    zip_file_path = os.path.join(TEMP_DIR, "neu_official.zip")
    try:
        download_success = download_file(NEU_DOWNLOAD_URL, zip_file_path)
        
        if download_success:
            # 解压数据集
            print("2. 解压数据集...")
            extract_zip(zip_file_path, TEMP_DIR)
            
            # 组织数据集
            print("3. 组织数据集...")
            # 查找解压后的数据集目录
            extracted_dir = None
            for root, dirs, files in os.walk(TEMP_DIR):
                for file in files:
                    if file.endswith('.jpg') and any(cls in file for cls in NEU_CLASSES.keys()):
                        extracted_dir = root
                        break
                if extracted_dir:
                    break
            
            if extracted_dir:
                organize_dataset(extracted_dir, DATASET_DIR, NEU_CLASSES)
                print("\n✅ NEU官方数据集下载并组织完成！")
            else:
                print("❌ 无法找到图像文件，可能下载的文件格式不正确")
                download_success = False
    except Exception as e:
        print(f"❌ 处理官方数据集时发生错误: {e}")
        download_success = False
    
    # 如果官方下载失败，尝试从GitHub下载
    if not download_success:
        print("\n4. 尝试从GitHub仓库克隆数据集...")
        github_dir = os.path.join(TEMP_DIR, "github_repo")
        try:
            download_success = download_from_github(GITHUB_REPO_URL, github_dir)
            
            if download_success:
                # 组织数据集
                print("5. 组织数据集...")
                organize_dataset(github_dir, DATASET_DIR, NEU_CLASSES)
                print("\n✅ GitHub数据集下载并组织完成！")
            else:
                print("❌ GitHub克隆失败")
        except Exception as e:
            print(f"❌ 处理GitHub数据集时发生错误: {e}")
            download_success = False
    
    # 如果GitHub下载失败，尝试从Kaggle下载
    if not download_success:
        print("\n6. 尝试从Kaggle下载真实NEU数据集...")
        zip_file_path = os.path.join(TEMP_DIR, "neu_kaggle.zip")
        try:
            download_success = download_from_kaggle(KAGGLE_URL, zip_file_path)
            
            if download_success:
                # 解压数据集
                print("7. 解压数据集...")
                extract_zip(zip_file_path, TEMP_DIR)
                
                # 组织数据集
                print("8. 组织数据集...")
                # 查找解压后的数据集目录
                extracted_dir = None
                for root, dirs, files in os.walk(TEMP_DIR):
                    for file in files:
                        if file.endswith('.jpg') and any(cls in file for cls in NEU_CLASSES.keys()):
                            extracted_dir = root
                            break
                    if extracted_dir:
                        break
                
                if extracted_dir:
                    organize_dataset(extracted_dir, DATASET_DIR, NEU_CLASSES)
                    print("\n✅ Kaggle数据集下载并组织完成！")
                else:
                    print("❌ 无法找到图像文件，可能下载的文件格式不正确")
                    download_success = False
        except Exception as e:
            print(f"❌ 处理Kaggle数据集时发生错误: {e}")
            download_success = False
    
    # 清理临时文件
    try:
        shutil.rmtree(TEMP_DIR)
        print("\n临时文件已清理")
    except Exception as e:
        print(f"\n清理临时文件失败: {e}")
    
    # 如果所有下载都失败，创建虚拟数据集
    if not download_success:
        print("\n=== 备选方案: 创建虚拟数据集 ===")
        print("由于所有下载尝试都失败，将创建包含模拟缺陷的虚拟数据集供测试使用")
        
        # 询问用户是否继续
        response = input("是否创建虚拟数据集？(y/n): ").lower()
        if response != 'y':
            print("已取消虚拟数据集创建")
            return
        
        create_dummy_data(DATASET_DIR)
        print("\n✅ 虚拟数据集已创建完成！")
    
    # 清理临时文件
    print("\n清理临时文件...")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    print("\n=== 任务完成 ===")
    print(f"数据集已保存到: {DATASET_DIR}")
    print("请检查dataset目录，确认图像文件已正确组织")

if __name__ == "__main__":
    main()