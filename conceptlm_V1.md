1. 先保留 baseline 主干 
tokenizer、embedding、position、transformer block、lm head、NTP loss 都先不变。checkpoint 也照常从 baseline 加载。                                                                                          
2. 在主干中间取 hidden state 选一个位置作为 encoder_hidden_states。最基础实现可以像本地 Pythia 配置一样： encoder_layers=0                                                                                                     decoder_layers=原总层数 也就是 concept 分支直接吃 embedding 后的 hidden；更稳一点也可以拆成前几层 encoder、后几层 decoder。                                                                                  
3. 把 token hidden 压成 concept hidden 每 chunk_size=4 个 token 合成一个 concept：
 meanpooling：对 4 个 token hidden 平均 得到形状大概是：token hidden:   [B, T, D] concept hidden: [B, T/4, D]                                                                                       
4. 加 VQ codebook，把 concept hidden 离散化 本地最基础配置是：vq_type: SimVQ codebook_size: 128 实现上按 attention head 切 hidden，每个 head 一个 codebook：concept hidden -> [B, C, num_heads, head_dim]每个 head 做 VQ -> concept id 这一步产生离散的 concept vocabulary。

5. 加 HLM 做 Next Concept Prediction加一个小 transformer，类似本地 special_layers=2。它输入当前 concept hidden，预测下一个 concept id：
concept_t -> predict concept_{t+1}
这就是论文里的 NCP。loss 可以有两种：
- CE：预测下一个 VQ index
- MSE：把预测出来的 concept vector 拟合下一个 concept hidden
本地基础脚本主要用：
--loss_type hlm_MSE_loss

6. 把预测 concept 注入回 token 路径HLM 预测出下一个 concept 后，再从 codebook 还原成连续向量。训练时建议用 softmax over codebook，比 hard_top1 梯度更顺：
--concept_merge_mode softmax
然后把 concept vector repeat 回 token 长度：
[B, T/4, D] -> repeat 4 次 -> [B, T, D]
最基础注入方式就是 raw add：
decoder_input = encoder_hidden_states + predicted_concept_hidden

7. decoder 和 lm head 继续原来的 token 预测
    后面照常：
    decoder -> final_norm -> lm_head -> next token CE

8. 总 loss
    训练时不是只训 NCP，而是三项一起：
    total_loss = token_CE + VQ_loss + NCP_loss
    其中：
    - token_CE 是 baseline 原来的 next-token loss
    - VQ_loss 约束 codebook 学得住
    - NCP_loss 训练 concept predictor