# LLaDA-V 提示词小样本（POPE-COCO 24题，Native 256/128/64）

| 组 | 提示 | 完成 | 画布 | EOS位置中位数 | 提前停止 | 预算耗尽 | 平均步数 | 平均秒 | 答案来源 | 准确率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| A | 当前：/think＋Final answer后缀 | 24/24 | 256 | 255 | 0 | 0 | 128.0 | 47.2 | final_answer:24 | 91.7% |
| B | /think，无后缀 | 24/24 | 256 | 255 | 0 | 0 | 128.0 | 46.8 | pope_official_first_sentence:24 | 50.0% |
| C | 原题，无标记无后缀 | 5/24 | 256 | 255 | 0 | 0 | 128.0 | 48.7 | pope_official_first_sentence:5 | 80.0% |
| D | /no_think＋Please answer yes or no. | 24/24 | 256 | 255 | 1 | 0 | 124.8 | 45.4 | pope_official_first_sentence:24 | 50.0% |
| A+eot | 当前：/think＋Final answer后缀；允许<|endoftext|>收尾 | 5/24 | 256 | 255 | 0 | 0 | 128.0 | 50.1 | final_answer:5 | 100.0% |
| L64_B64 | Native 64/32，/think无后缀 | 24/24 | 64 | 63 | 0 | 0 | 32.0 | 10.8 | leading_yes_no:24 | 83.3% |
| L64_A64 | Native 64/32，/think＋Final answer后缀 | 24/24 | 64 | 63 | 0 | 0 | 32.0 | 10.9 | final_answer:24 | 91.7% |
| L64_E64 | CDAPC L64/I32 K4，/think＋后缀 | 24/24 | 64 | 63 | 0 | 9 | 29.2 | 18.2 | final_answer:15，generation_failed:9 | 58.3% |

示例（每组前两题开头）：
- A: To determine if there is a handbag in the image, let's break down the observation step by step:  1. **Observation of the Image**: The image shows a group of people sitting around a table. The table is covered with variou
- A: To determine if there are skis in the image, we need to carefully examine the objects and equipment present.  1. **Observation of the Image**: The image shows a group of people standing on a snowy slope. They are dressed
- B: To determine if there is a handbag in the image, we need to carefully examine the objects present on the table.   1. **Observation of the Table**: The table in the image has several items on it. including a    - A plate 
- B: To determine if there are skis in the image, we need to carefully examine the objects and equipment present.   1. **Observation of the Image**: The image shows a group of people standing on a snowy slope. They are wearin
- C: To determine if there is a handbag in the image, we need to carefully examine the objects present on the table.   1. **Observation of the Table**: The table in the image has several items on it. including a    - A plate 
- C: To determine if there are skis in the image, we need to carefully examine the objects present in the scene.   1. **Observation of the Image**: The image shows a group of people standing on a snowy slope. They are dressed
- D: To determine if there is a handbag in the image, we need to carefully examine the objects present on the table.   1. **Observation of the Table**: The table in the image has several items on it. including a    - A plate 
- D: To determine if there are skis in the image, we need to carefully examine the objects present in the scene.   1. **Observation of the Image**: The image shows a group of people standing on a snowy slope. They are dressed
- A+eot: To determine if there is a handbag in the image, let's break down the observation step by step:  1. **Observation of the Image**: The image shows a group of people sitting around a table. The table is covered with variou
- A+eot: To determine if there are skis in the image, we need to carefully examine the objects and equipment present.  1. **Observation of the Image**: The image shows a group of people standing on a snowy slope. They are dressed
- L64_B64: Yes, there is a handbag in the image. It is located on the table to the right of the person in the foreground, The handbag appears to be a dark color, possibly leather, and is of a medium size, although the brand and bra
- L64_B64: Yes, there are skis in the image. The individuals are equipped with skis, which are attached to their boots and are positioned on the snow. The skis are of various lengths and designs, reflecting the different styles and
- L64_A64: To determine if there is a handbag in the image, we need to carefully examine the objects present on the table. Upon close inspection, we can see a black object placed on the table near the center, which matches the appe
- L64_A64: To determine if there are skis in the image, we need to carefully observe the scene depicted. The image shows a group of people standing on a snowy surface. Each person is wearing a pair of skis attached to their feet. T
- L64_E64: To determine if there is a handbag in the image, we need to carefully examine the objects present on the table. Upon close inspection, we can see a black handbag placed on the table near the person in the foreground. Thi
- L64_E64: To determine if there are skis in the image, we need to carefully observe the scene depicted. The image shows a group of people standing on a snowy slope. Each person has a pair of skis attached to their feet. Therefore,
