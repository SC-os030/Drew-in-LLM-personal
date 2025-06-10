一、任务目标：将自然语言描述转换为能够精确渲染对应图像的SVG代码。与传统的文本到图像生成（通常产生光栅图像）不同，本次竞赛要求生成结构化的SVG矢量图形。
整体解决方案架构
流程图：
```
graph TD
    A[文本提示输入] --> B{Prompt工程};
    B --> C[SDXL + LoRA图像生成];
    C --> D[高质量位图];
    D --> E{多次尝试循环};
    E -- 生成 --> F[位图到SVG转换模块];
    F --> G[候选SVG];
    G --> H[SVG合规性检查与大小控制];
    H --> I[基于美学评分的评估与选择];
    I -- 最佳SVG --> J[最终SVG后处理];
    J --> K[输出最终SVG代码];
    E -- 达到尝试次数 --> I;
```
二、流程概述
1.Prompt工程（Prompt Engineering）:对输入的原始文本提示进行增强，通过添加预定义的前缀 (Prefix) 和后缀 (Suffix)，以及设置反向提示 (Negative Prompt)，来引导图像生成模型产生更符合期望风格（如“矢量风”、“卡通”）和质量的图像。
2.图像生成（Image Generation）:利用Stable Diffusion XL (SDXL)模型作为基础，同时为了加速生成并优化输出风格，本次项目集成了SDXL Lightning UNet权重，允许在极少数的推理步数（如4-9步）下获得高质量的图像。更为关键的是加载了Doctor Diffusion's Controllable Vector" LoRA 权重。这个LoRA经过特殊训练，可以引导SDXL生成具有更清晰边缘，平坦颜色区域和简化细节的图像，这些步骤适合后序的矢量化处理。
