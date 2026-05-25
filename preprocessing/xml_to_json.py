import xml.etree.ElementTree as ET
import json
import os
import glob
import argparse

def parse_ddi_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    document_id = root.attrib.get('id', '')
    
    samples = []
    
    for sentence in root.findall('sentence'):
        sentence_id = sentence.attrib.get('id', '')
        text = sentence.attrib.get('text', '')
        
        entities = {}
        for entity in sentence.findall('entity'):
            e_id = entity.attrib.get('id')
            char_offset = entity.attrib.get('charOffset')
            # Handle possible disjoint offsets like "10-15;20-25"
            first_offset = char_offset.split(';')[0]
            start_str, end_str = first_offset.split('-')
            char_start = int(start_str)
            char_end = int(end_str) + 1 # +1 to match string slicing e.g. [char_start:char_end]
            
            entities[e_id] = {
                'id': e_id,
                'text': entity.attrib.get('text', ''),
                'type': entity.attrib.get('type', ''),
                'char_start': char_start,
                'char_end': char_end
            }
            
        for pair in sentence.findall('pair'):
            p_id = pair.attrib.get('id')
            e1_id = pair.attrib.get('e1')
            e2_id = pair.attrib.get('e2')
            interaction = pair.attrib.get('interaction', 'false').lower().strip()
            
            if e1_id in entities and e2_id in entities:
                sample = {
                    'id': p_id,
                    'sentence_id': sentence_id,
                    'document_id': document_id,
                    'text': text,
                    'h_entity': entities[e1_id],
                    't_entity': entities[e2_id],
                    'relation': interaction
                }
                samples.append(sample)
                
    return samples

def process_directory(input_dir, output_file):
    all_samples = []
    xml_files = glob.glob(os.path.join(input_dir, '*.xml'))
    
    for xml_file in xml_files:
        try:
            samples = parse_ddi_xml(xml_file)
            all_samples.extend(samples)
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
    print(f"Processed {len(xml_files)} files. Extracted {len(all_samples)} relations.")
    print(f"Output saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert DDI XML corpus to JSON")
    parser.add_argument('--input_dir', type=str, default=None, help="Directory containing XML files")
    parser.add_argument('--output_file', type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    if args.input_dir and args.output_file:
        process_directory(args.input_dir, args.output_file)
    else:
        print("Không truyền tham số, tự động build dữ liệu từ thư mục raw...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # --- ĐÃ SỬA LẠI THEO ĐÚNG CẤU TRÚC THƯ MỤC MỚI ---
        raw_in = os.path.join(project_root, "data", "raw")
        raw_out = os.path.join(project_root, "data", "dataset.json") 
        
        if os.path.exists(raw_in):
            process_directory(raw_in, raw_out)
        else:
            print(f"[LỖI] Không tìm thấy thư mục: {raw_in}")