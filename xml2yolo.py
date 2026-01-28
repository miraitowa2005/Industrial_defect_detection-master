import xml.etree.ElementTree as ET
import os
import glob

classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(xml_path, output_path):
    with open(xml_path, encoding='utf-8') as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text if obj.find('difficult') is not None else '0'
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (
                    float(xmlbox.find('xmin').text),
                    float(xmlbox.find('xmax').text),
                    float(xmlbox.find('ymin').text),
                    float(xmlbox.find('ymax').text),
                )
                bb = convert((w, h), b)
                out_file.write(f"{cls_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")

def main():
    root_dir = r'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset\NEU-DET'
    for folder in ['train', 'validation']:
        xml_dir = os.path.join(root_dir, folder, 'annotations')
        label_dir = os.path.join(root_dir, folder, 'labels')
        os.makedirs(label_dir, exist_ok=True)
        xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
        for xml_file in xml_files:
            # 根据XML中的类别建立对应的子目录，确保与images的子目录结构一致
            with open(xml_file, encoding='utf-8') as in_file:
                tree = ET.parse(in_file)
                root = tree.getroot()
                obj = root.find('object')
                cls = obj.find('name').text if obj is not None else None
            sub_dir = os.path.join(label_dir, cls) if cls else label_dir
            os.makedirs(sub_dir, exist_ok=True)
            file_name = os.path.basename(xml_file).replace('.xml', '.txt')
            out_path = os.path.join(sub_dir, file_name)
            convert_annotation(xml_file, out_path)
    print('XML to YOLO conversion done.')

if __name__ == '__main__':
    main()
