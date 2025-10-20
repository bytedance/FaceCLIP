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


import torch

def attn_ref(q, k, v, b, sm_scale, dropout_p=0.0, causal=False, upcast=False):
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
        if b is not None:
            b = b.float()

    if b is not None:
        if (b.shape[0] != q.shape[0]) or (b.shape[1] != q.shape[1]):
            b = b.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])

    ms = torch.arange(q.shape[2], device=q.device).unsqueeze(-1)
    ns = torch.arange(k.shape[2], device=q.device)

    p = torch.matmul(q, k.transpose(2, 3))
    p *= sm_scale
    if b is not None:
        p += b

    if causal:
        p = torch.where(ms + k.shape[2] - q.shape[2] >= ns, p, float("-inf"))

    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    if dropout_p > 0.0:
        p = torch.dropout(p, dropout_p, train=True)

    ref_out = torch.matmul(p, v)
    return ref_out
