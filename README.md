

![UE封面图 正式版](https://github.com/ZHO-ZHO-ZHO/ComfyUI-UltraEdit-ZHO/assets/140084057/40d86c06-b6ee-4a72-a25b-fc470fa3a424)


# UltraEdit in ComfyUI

Unofficial implementation of [UltraEdit](https://github.com/HaozheZhao/UltraEdit)（Diffusers） for ComfyUI


![screenshot-20240710-033335](https://github.com/ZHO-ZHO-ZHO/ComfyUI-UltraEdit-ZHO/assets/140084057/b10be119-85a4-435b-b68c-cb5dc4b4f3b1)



## 项目介绍 | Info

- 对 [UltraEdit](https://github.com/HaozheZhao/UltraEdit) 的非官方 Diffusers 版实现

- UltraEdit：基于 SD3 Medium 的 图像编辑模型
    - 允许仅通过提示词实现指定内容的图像编辑，且能保持风格一致性
    - 模型同时支持 全局编辑 和 区域蒙版编辑
  
- 版本：V1.0 同时支持 本地模型加载（适合本地）和 自动下载模型（适合云端）



## 节点说明 | Features

- UltraEdit 模型加载 | 🏕️UltraEdit Model
    - 🏕️UltraEdit Model(local)：加载本地模型，需将 [BleachNick/SD3_UltraEdit_w_mask](https://huggingface.co/BleachNick/SD3_UltraEdit_w_mask/tree/main) 中的所有模型和文件手动下载放入 `/ComfyUI/models/ultraedit` 中
    - 🏕️UltraEdit Model(auto)：会自动下载模型，请保持网络畅通
    
- UltraEdit 生成 | 🏕️UltraEdit Generation
    - pipe：接入模型
    - image：接入图片
    - mask：非必要项，接入蒙版（需图片格式，可通过 convert mask to image 节点转换）
    - positive：正向提示词
    - negative：负向提示词
    - step：步数，默认 50 步（模型是 512*512）
    - image_guidance_scale：图像相关度，默认为 1.5
    - text_guidance_scale：文本引导相关度，默认为 7.5
    - seed：种子

![screenshot-20240710-183111](https://github.com/ZHO-ZHO-ZHO/ComfyUI-UltraEdit-ZHO/assets/140084057/2d990858-a8bc-4865-8b90-5e6d9c7cb177)



## 使用注意 | Attention

- 输入图像尺寸需是 4 的倍数

- 如果得到的输出是 512 张图而非单张图像，那需要手动将 UltraEdit.py 文件最后的 `output_t = output_t.squeeze(0)` 删除，我自己的测试均未出现这个问题，目前还不清楚是啥原因（估计还是 diffusers 问题）



## 安装 | Install

- **注意**：此项目需要特殊的 diffusers 版本，推荐使用虚拟环境或云（避免冲突），可直接通过 requirements 自动安装依赖，也可手动安装：`pip install git+https://github.com/HaozheZhao/UltraEdit.git@main#subdirectory=diffusers`

- 推荐使用管理器 ComfyUI Manager 安装

- 手动安装：
    1. `cd custom_nodes`
    2. `git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-UltraEdit-ZHO.git`
    3. `cd custom_nodes/ComfyUI-UltraEdit-ZHO`
    4. `pip install -r requirements.txt`
    5. 重启 ComfyUI


## 工作流 | Workflows

V1.0

  - 加载本地模型：[V1.0 UltraEdit global & mask edit（local）](https://github.com/ZHO-ZHO-ZHO/ComfyUI-UltraEdit-ZHO/blob/main/UltraEdit%20Workflows/V1.0%20UltraEdit%20global%20%26%20mask%20edit%EF%BC%88local%EF%BC%89%E3%80%90Zho%E3%80%91.json)
  - 自动下载模型：[V1.0 UltraEdit global & mask edit（auto）](https://github.com/ZHO-ZHO-ZHO/ComfyUI-UltraEdit-ZHO/blob/main/UltraEdit%20Workflows/V1.0%20UltraEdit%20global%20%26%20mask%20edit%EF%BC%88auto%EF%BC%89%E3%80%90Zho%E3%80%91.json)

    ![screenshot-20240710-032255](https://github.com/ZHO-ZHO-ZHO/ComfyUI-UltraEdit-ZHO/assets/140084057/9651520f-59a2-45ce-ab20-7335dd839007)




## 更新日志

- 20240711

  修复pil库版本低的问题，已适配最新库

  新增使用注意

  新增 本地模型加载 工作流，分为 本地加载/自动下载 两种

- 20240710

  V1.0 同时支持 全局编辑 和 局部蒙版编辑，同时支持 本地模型加载（适合本地）和 自动下载模型（适合云端）

  创建项目
  

## Stars 

[![Star History Chart](https://api.star-history.com/svg?repos=ZHO-ZHO-ZHO/ComfyUI-UltraEdit-ZHO&type=Date)](https://star-history.com/#ZHO-ZHO-ZHO/ComfyUI-UltraEdit-ZHO&Date)


## 关于我 | About me

📬 **联系我**：
- 邮箱：zhozho3965@gmail.com
- QQ 群：839821928

🔗 **社交媒体**：
- 个人页：[-Zho-](https://jike.city/zho)
- Bilibili：[我的B站主页](https://space.bilibili.com/484366804)
- X（Twitter）：[我的Twitter](https://twitter.com/ZHOZHO672070)
- 小红书：[我的小红书主页](https://www.xiaohongshu.com/user/profile/63f11530000000001001e0c8?xhsshare=CopyLink&appuid=63f11530000000001001e0c8&apptime=1690528872)

💡 **支持我**：
- B站：[B站充电](https://space.bilibili.com/484366804)
- 爱发电：[为我充电](https://afdian.net/a/ZHOZHO)


## Credits

[UltraEdit](https://github.com/HaozheZhao/UltraEdit)
