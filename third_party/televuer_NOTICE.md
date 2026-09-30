# televuer (MIT) -- notice for adapted code

`source/dropbear_wbc/teleop/frames.py` adapts constants and the head-relative wrist transform from
unitreerobotics/televuer, `src/televuer/tv_wrapper.py` (github.com/unitreerobotics/televuer, commit
766de45e74373ae0ea66321d942ce538385655a5). `source/dropbear_wbc/teleop/devices.py` (`VuerXRSource`) follows the
Vuer event handling of `src/televuer/televuer.py`. No file was copied verbatim.

```
MIT License

Copyright (c) 2025, Unitree Robotics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

xr_teleoperate (unitreerobotics/xr_teleoperate @ 817fb00c) was used as the design reference for the IK cost,
the arm-controller gains/rates and the recording layout; its license:

```
Copyright [2025] [HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------

This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES:

1) https://github.com/OpenTeleVision/TeleVision
2) https://github.com/dexsuite/dex-retargeting
3) https://github.com/vuer-ai/vuer
4) https://github.com/stack-of-tasks/pinocchio
5) https://github.com/casadi/casadi
6) https://github.com/meshcat-dev/meshcat-python
7) https://github.com/zeromq/pyzmq
8) https://github.com/Dingry/BunnyVisionPro
9) https://github.com/unitreerobotics/unitree_sdk2_python

------------------
```
