# YOLO8+ESP32

`main.py` 用于读取 ESP32 视频流，先用 YOLO 检测配置文件中指定的类别，再在 `person` 框内做人脸识别并标注 `admin`。

## 运行

```powershell
python YOLO8+esp32\main.py
```

常用命令行参数：

- `--conf`：YOLO 检测置信度阈值，默认 `0.2`
- `--yolo-label-classes`：覆盖配置文件中的标签列表，例：`person`、`person,bottle`、`none`

说明：

- 如果传了 `--yolo-label-classes`，命令行优先。
- 如果不传，程序默认读取同目录下的 `label_config.json`。

## 配置文件

配置文件路径：`YOLO8+esp32\label_config.json`

当前支持的字段：

```json
{
  "yolo_label_classes": ["person"]
}
```

含义：

- `yolo_label_classes`：YOLO 实际检测并显示的类别列表
- 例如 `["person"]` 表示只检测 `person`
- 例如 `[]` 表示不检测任何 YOLO 类别

注意：

- 现在的人脸识别依赖 `person` 检测结果做级联。
- 如果配置里不包含 `person`，`admin` 人脸识别也不会正常工作。

## 人脸识别参数

这些参数当前定义在 `main.py` 中：

- `FACE_TOLERANCE`：单个最佳样本距离阈值
- `FACE_MEAN_TOLERANCE`：同一身份前 `FACE_TOP_K` 个样本的平均距离阈值
- `FACE_TOP_K`：参与平均距离计算的样本数
- `FACE_RECOGNITION_SCALE`：实时人脸识别缩放比例
- `MIN_PERSON_BOX_SIZE`：参与人脸识别的最小 `person` 框尺寸
- `PERSON_BOX_PADDING_RATIO`：对 `person` 框扩边的比例

当前默认值以 `main.py` 为准。

## 人脸库

支持两种人脸样本来源：

- `YOLO8+esp32\known_face.jpg`
- `YOLO8+esp32\known_faces\admin\*.jpg`

建议在 `known_faces\admin\` 下放多张清晰样本图，覆盖正脸、轻微侧脸和不同光照。

## 当前处理流程

1. 读取 ESP32 视频帧
2. YOLO 只检测配置文件中指定的类别
3. 从检测出的 `person` 框里裁剪区域
4. 在裁剪区域内执行 `face_recognition`
5. 在画面中绘制 YOLO 框和人脸识别结果

## 验证

修改代码后可用以下命令做语法检查：

```powershell
python -m py_compile YOLO8+esp32\main.py
```
