# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import time


class TimeCounter:
    def __init__(self) -> None:
        pass

    def clear(self):
        self.timedict = {}
        self.basetime = time.perf_counter()

    def timeit(self, name):
        nowtime = time.perf_counter() - self.basetime
        self.timedict[name] = nowtime
        self.basetime = time.perf_counter()


class TimeHolder:
    def __init__(self) -> None:
        self.timedict = {}

    def update(self, _timedict: dict):
        for k, v in _timedict.items():
            if k not in self.timedict:
                self.timedict[k] = AverageMeter(name=k, val_only=True)
            self.timedict[k].update(val=v)

    def final_res(self):
        return {k: v.avg for k, v in self.timedict.items()}

    def __str__(self):
        return json.dumps(self.final_res(), indent=2)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = "{name} {val" + self.fmt + "}"
        else:
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
