for img in *.jpg ; do
width=$(identify -format "%w" "$img")
height=$(identify -format "%h" "$img")
cat <<-EOF > ${img%.jpg}.json
{
  "imageData": null,
  "flags": {
    "p": false
  },
  "fillColor": [
    255,
    0,
    0,
    128
  ],
  "lineColor": [
    0,
    255,
    0,
    128
  ],
  "imagePath": "$img",
  "imageHeight": $height,
  "imageWidth": $width,
  "shapes": []
}
EOF
done
