# 2021 Tianchi Warm Up Competition (Fabric Defects)  

### 正当高三寒假，经过了两周的探索，第一次参加天池比赛并使用Docker提交后有以下经验和大家分享。如有错误，一定及时修改！！！

这次热身赛原本是[**2019广东工业智造创新大赛【赛场一】**](https://tianchi.aliyun.com/competition/entrance/231748/introduction)，所以在[论坛](https://tianchi.aliyun.com/competition/entrance/231748/forum)中有非常多的优秀的Baseline供我们（小白）参考，我的Baseline就是融合了论坛中的大部分模型，加以修改而来，感谢他们的开源精神！

## Baseline

- 框架是使用[mmdetection](https://github.com/open-mmlab/mmdetection)，basic-baseline是[Cascade R-CNN X-101-32x4d-FPN](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn)

- 安装和训练[官方文档](https://mmdetection.readthedocs.io/en/latest/index.html)十分详细了，一步一步跟着来没什么问题。[Docker配置](https://mmdetection.readthedocs.io/en/latest/get_started.html#another-option-docker-image)也很全，但我根据CSDN大佬修改过的Dockerfile再次修改，加速了国内pip安装。

- Baseline是来自论坛中的各支队伍的单模混合起来的！我的config文件就在同级目录里面，具体修改细节已注释。

- 当batch size = 2时，显存需要15.9G，需要注意！


## mmdetection的Dockerfile（感谢CSDN的大佬）

```
ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html -i https://pypi.douban.com/simple/
RUN conda clean --all

#RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
# 前面这个mmdetection/含有自己的调试好的代码，后面这个是docker环境里/workspace/mmdetection这个文件夹
ADD mmdetection/ mmdetection/
WORKDIR mmdetection/
ENV FORCE_CUDA="1"
RUN apt update && apt install unzip nano
RUN pip install matplotlib -i https://pypi.douban.com/simple/
RUN pip install -r requirements/build.txt -i https://pypi.douban.com/simple/
RUN pip install --no-cache-dir -e .
RUN pip install icecream

RUN apt-get update
RUN apt-get install unzip nano curl
RUN cp /workspace/mmdetection/run.sh /run.sh
RUN chmod 777 /run.sh
```
安装curl，如果没有安装，提交后会报错，但不浪费提交次数。猜测是用来获取 result.json

## Yolov5的Docker可以直接拉取（对于本次比赛不推荐使用）
Yolov5官方Docker文档链接：https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart
不推荐原因：
1. 主流的视觉检测框架有三个（我了解到的）
	1. facebook的Detectron2: https://github.com/facebookresearch/detectron2
	2. 港中大和商汤的mmdetection(2.0): https://github.com/open-mmlab/mmdetection
	3. Yolo(v5): https://github.com/ultralytics/yolov5
2. 第一次尝试Yolov5属于One-stage，训练150 epochs：提交结果：acc: 72.4872；mAP: 8.0260，直接抛弃。【使用yolov5s.pt作为基础模型，没用调整除batch size, img size 和epoch的任何参数】
	* One-stage速度快，准确率低
	* Two-stage准确率高，速度略慢
	* -->本次检测应该用Two-stage的模型

## 训练模型时的问题

1. 我的笔记本是Thinkpad T490，独显是Nvidia GeForeMX250，2G显存，性能弱爆，跑Yolov5可以的，mmdetection就Out of Memory，果断开始租GPU服务器（在氪金的道路上越走越远。。。）

2. mmdetection训练第一轮的时候，显存会逐渐增加。如果第一个batch的时候，显存占用超过**30%**，估计撑不到第一个epoch结束就会**Out of Memory**。啪，时间和钱都没了:(。【第一个batch使用26%的显存，epoch中间的时候就94%。于是开始祈祷了，幸好撑到最后！】

## 提交的问题（Docker自学吧<s>，没什么说的</s>）

### Build Docker
```
cd 到存放Dockerfile的目录
docker build -t mmdetection .
```
### Docker Command
```
docker run -it -v /data:/tcdata -w / -p 8000:22 --ip=host 你的镜像名字 /bin/bash
解释：
	-v 宿主机目录:虚拟机目录（简单一点讲就是共享文件夹）
	-w / （工作目录是根目录，不是/root，是 / 根目录，你想啥目录都行）
	-p 宿主机端口:虚拟机端口 （端口转发）
	--ip=host 我是服务器上的docker，为能ssh连接虚拟机，所以一定要写这个
	/bin/bash 持久化（意思能一直开着）
	更详细的可以看菜鸟教程: https://www.runoob.com/docker/docker-run-command.html
修改完容器，打一个镜像
docker ps -l # 查看在运行的容器（Container）
docker commit 容器ID(前3个字符或更多) 新的镜像名字:版本

接下来就根据腾讯云或者阿里云镜像的提示，push到自己的命名仓库里面后就可以提交了。
```

## 吐槽部分

为什么学历最低只有高中，像我高中在读，初中学历为啥不在列表？

列表的学校也只有大学，高中不叫学校？奇奇怪怪的
