```shell
mosaic_detection_dataset_dir=datasets/mosaic_detection

val_samples_count=500
python make_video_dataset_for_mosaic.py --input-root $mosaic_detection_dataset_dir --output-root $mosaic_detection_dataset_dir
python lada/yolo/convert-mask-images-to-coco.py --images-dir $mosaic_detection_dataset_dir/img --masks-dir $mosaic_detection_dataset_dir/mask --output-file $mosaic_detection_dataset_dir/coco.json
python lada/yolo/convert-coco-to-yolo.py --coco-file $mosaic_detection_dataset_dir/coco.json --yolo-labels-dir $mosaic_detection_dataset_dir/mask_yolo

mkdir -p $mosaic_detection_dataset_dir/yolo/{train,val}/{labels,images}
find $mosaic_detection_dataset_dir/img -type f -name "*.jpg"  | sort -R | head -n $val_samples_count | while read img_path ; do 
    mask_path=$mosaic_detection_dataset_dir/mask_yolo/$(basename ${img_path%.jpg}.txt)
    mv $img_path $mosaic_detection_dataset_dir/yolo/val/images
    mv $mask_path $mosaic_detection_dataset_dir/yolo/val/labels
done
find $mosaic_detection_dataset_dir/img -type f -name "*.jpg" | while read img_path ; do 
    mask_path=$mosaic_detection_dataset_dir/mask_yolo/$(basename ${img_path%.jpg}.txt)
    mv $img_path $mosaic_detection_dataset_dir/yolo/train/images
    mv $mask_path $mosaic_detection_dataset_dir/yolo/train/labels
done

rmdir $mosaic_detection_dataset_dir/img
rmdir $mosaic_detection_dataset_dir/mask_yolo
rm -r $mosaic_detection_dataset_dir/mask
rm $mosaic_detection_dataset_dir/coco.json
mv $mosaic_detection_dataset_dir/yolo/{train,val} $mosaic_detection_dataset_dir/
rmdir $mosaic_detection_dataset_dir/yolo
```

