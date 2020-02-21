import json

minival_json_path = 'datasets/coco/annotations/instances_minival2014.json'
microval_json_path = 'datasets/coco/annotations/instances_microval2014.json'

# Reading data back
with open(minival_json_path, 'r') as f:
    data = json.load(f)

img_ids = [img['id'] for img in data['images']]
# unique_sorted_img_ids = sorted(list(set(img_ids)))

# ann_mentioned_img_ids = [ann['image_id'] for ann in data['annotations']]
# unique_sorted__ann_img_ids = sorted(list(set(ann_mentioned_img_ids)))

# missed_img_ids = [img_id for img_id in unique_sorted_img_ids if img_id not in unique_sorted__ann_img_ids]



micro_num = 100
wanted_img_ids = img_ids[:micro_num]
micro_images = [img for img in data['images'] if img['id'] in wanted_img_ids]
micro_annotations = [ann for ann in data['annotations'] if ann['image_id'] in wanted_img_ids]
micro_data = data.copy()
micro_data['images'] = micro_images
micro_data['annotations'] = micro_annotations

# Writing JSON data
with open(microval_json_path, 'w') as f:
    json.dump(micro_data, f)


