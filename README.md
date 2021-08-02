# Quadrilateral_Corner_Extraction
Ubuntu 18.0
Opencv 3.2.0

我们提取图左边的四边形。
先用Otsu()算出其类间最大阈值，将其转换为黑白图像。
设置并获取图像中的有效位置。
获取有效位置中的线段。
通过线段算出交点从而获得角点。
