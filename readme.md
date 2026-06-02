# MIDI Wrong-Note Detection with Transformer and BiGRU

## Current Project Status

This project has moved beyond the original BiGRU prototype. The main experimental line is now a Transformer-based MIDI wrong-note detector with synthetic error generation, chord/scale/degree features, melodic-theory features, precision-first calibration, and post-processing/reranking experiments.

The original BiGRU notes are kept below for history, but the current checkpoints and experiments should be read from:

- `experiments.md`: chronological experiment log and iteration path.
- `training_logs\*.log`: stdout logs for each long run.
- `training_logs\*.err.log`: stderr/progress logs.
- `checkpoints\*.pt`: saved model checkpoints.

## Important 40-Epoch Baseline

The previously important 40-epoch run used the BiGRU/default model line with theory/scheduler settings:

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new

E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe -u -m midi_error_detector.train `
  --data-root "E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0" `
  --clean-epochs 1 `
  --epochs 40 `
  --batch-size 16 `
  --window-size 256 `
  --train-error-rate 0.15 `
  --error-rate 0.08 `
  --det-threshold 0.3 `
  --det-pos-weight 3.0 `
  --kind-class-weights 1 6 4 `
  --threshold-sweep 0.2 0.25 0.3 0.35 0.4 0.5 `
  --save-metric task_score `
  --lr-patience 4 `
  --lr-factor 0.5 `
  --lr-threshold 0.002 `
  --num-workers 0 `
  --output checkpoints\bigru_wrong_note_theory_scheduler_taskscore.pt
```

Final epoch 40 test metrics:

- `task_score=0.774284316890155`
- fixed `det_threshold=0.3`: precision `0.4884`, recall `0.8842`, F1 `0.6293`
- best threshold `0.5`: precision `0.5994`, recall `0.8343`, F1 `0.6976`
- `replace_pitch_top3=0.8500`
- `replace_kind_acc=0.8520`
- `delete_kind_acc=0.9383`

This run had high recall but low precision, which is why the later work shifted toward low-error-rate evaluation, precision-first fine-tuning, and high-confidence post-processing.

## Transformer 40-Epoch Version

The Transformer version keeps the same wrong-note task but changes the sequence encoder from BiGRU to self-attention:

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new

E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe -u -m midi_error_detector.train `
  --model transformer `
  --data-root "E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0" `
  --clean-epochs 1 `
  --epochs 40 `
  --batch-size 8 `
  --window-size 256 `
  --num-layers 4 `
  --transformer-d-model 192 `
  --transformer-heads 4 `
  --transformer-ffn-dim 512 `
  --train-error-rate 0.15 `
  --error-rate 0.08 `
  --det-threshold 0.3 `
  --det-pos-weight 3.0 `
  --kind-class-weights 1 6 4 `
  --threshold-sweep 0.2 0.25 0.3 0.35 0.4 0.5 `
  --save-metric task_score `
  --lr-patience 4 `
  --lr-factor 0.5 `
  --lr-threshold 0.002 `
  --num-workers 0 `
  --output checkpoints\transformer_wrong_note_taskscore.pt
```

Current Transformer-family checkpoints:

- `checkpoints\transformer_chord_degree_taskscore.pt`: Transformer with chord/degree features.
- `checkpoints\transformer_precision_finetune.pt`: low-error-rate precision-first fine-tune. Best result: precision-task score `0.8106`, precision `0.8753`, recall `0.5006` at threshold `0.97`.
- `checkpoints\transformer_melodic_theory_precision.pt`: melodic theory feature version. Best result: precision-task score `0.8108`, precision `0.8693`, recall `0.5102` at threshold `0.97`.
- `checkpoints\transformer_theory_weighted_recall.pt`: current theory-weighted recall run, intended to recover recall while keeping precision high.

The current target is not only a higher `task_score`; the practical target is to push both precision and recall toward 80%+ under realistic sparse wrong-note conditions.
这个项目提供一个面向 **AI 识谱后处理** 的 PyTorch 原型：输入已经转成 MIDI 的演奏结果，模型在 note-level 上判断哪些音可能是错音，并预测应替换成的正确 MIDI pitch。

## 数据集确认

MAESTRO 官网（`http://g.co/magenta/maestro-dataset` 会跳转到 Magenta 页面）说明：

- MAESTRO 是约 200 小时的钢琴演奏 MIDI/音频数据集。
- 数据包含 MIDI 与 WAV，以及 CSV/JSON 元数据。
- 元数据字段包括 `split`、`midi_filename`、`duration` 等。
- 官方提供 train/validation/test split，避免同一曲目跨 split 出现。
- 若只训练 MIDI 后处理模型，优先下载 `maestro-v3.0.0-midi.zip`，约 56 MB；无需下载 101 GB 的完整音频包。

## 建模思路

MAESTRO 本身是干净/正确的演奏数据，因此训练时在线合成错误：

1. `neighbor`：把正确音替换成相邻半音，模拟误触隔壁黑键/白键。
2. `nearby`：把正确音随机替换成附近若干半音内的音，模拟弹错到周围音。
3. `nearby_plus_touch`：先弹错附近音，同时额外加入一个邻近误触音，模拟“按错 + 误触”。

模型结构是双向 GRU：

- 输入：每个 note 的 pitch、velocity、duration、相邻 onset 间隔、pitch interval、onset 小数周期等特征。
- Head 1：二分类，输出该 note 是错音的概率。
- Head 2：三分类动作，输出该 note 应该 `keep`、`replace` 还是 `delete`。
- Head 3：128 类 pitch 分类；如果动作是 `replace`，可以取 top1 或 top3 候选正确 MIDI pitch。

## 安装

Linux/macOS 示例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

如果使用你已经建好的 Windows venv，可以在项目代码目录中直接调用该 venv 的 Python。你现在把代码放进了 `code_new`，所以推荐这样运行：

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new
E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe -m pip install -e .
```

如果你后来把 venv 也复制/新建到了 `code_new\venv`，则可以改用：

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new
.\venv\Scripts\python.exe -m pip install -e .
```

也可以直接安装依赖：

```bash
python -m pip install -r requirements.txt
```

项目同时提供 `setup.py` 和 `pyproject.toml`：`setup.py` 是传统 setuptools 入口，方便课程项目或旧工具识别；`pyproject.toml` 是现代 Python 打包配置。平时你只需要在 `code_new` 根目录运行 `python -m pip install -e .`，pip 会自动使用这些配置安装 `src\midi_error_detector` 包。

## 关于 `pip install -e .` 的输出

如果你看到类似下面的输出：

```text
Successfully built midi-error-detector
Installing collected packages: colorama, tqdm, midi-error-detector
Successfully installed colorama-0.4.6 midi-error-detector-0.1.0 tqdm-4.67.3
```

这不表示下载了一个已经训练好的模型，也不表示训练已经完成。它表示 pip 做了两件事：

1. 把你当前 `code_new` 目录里的本地源码，根据 `setup.py`/`pyproject.toml` 打包安装成名为 `midi-error-detector` 的 Python 包。
2. 安装运行代码需要的依赖，例如 `tqdm`、`colorama` 等。

真正的模型训练只有在你运行 `python -m midi_error_detector.train ...` 后才会开始。训练完成后，才会在 `--output` 指定的位置生成 checkpoint，例如 `checkpoints\bigru_wrong_note.pt`。

## 文件名和目录结构

如果你手动把代码整理到 `E:\downloads\桌面\dku\CS309\project\code_new`，建议保留下面这些有效文件名和相对路径；不要把所有 `.py` 都平铺到同一个文件夹，否则 `python -m midi_error_detector.train` 这种包运行方式会找不到模块。

```text
E:\downloads\桌面\dku\CS309\project\code_new\
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── scripts\
│   └── train_windows.ps1
└── src\
    └── midi_error_detector\
        ├── __init__.py
        ├── data.py
        ├── model.py
        ├── train.py
        └── predict.py
```

这几个 `.py` 文件的作用分别是：

- `src\midi_error_detector\__init__.py`：告诉 Python 这是一个包。
- `src\midi_error_detector\data.py`：读取 MAESTRO MIDI、造错音数据、生成训练特征。
- `src\midi_error_detector\model.py`：定义 BiGRU 模型和 loss。
- `src\midi_error_detector\train.py`：训练入口。
- `src\midi_error_detector\predict.py`：推理入口，加载 checkpoint 后检测/纠错单个 MIDI。

`train_windows.ps1` 不是 Python 文件，它是 Windows PowerShell 脚本，作用只是把一长串训练命令写成一个可双击/可命令行调用的快捷脚本；不用它也完全可以，直接运行下面 README 里的 `python -m midi_error_detector.train ...` 命令即可。

## 准备 MAESTRO MIDI 数据

下载并解压 MIDI-only 数据包，使目录类似：

```text
/path/to/maestro-v3.0.0/
├── maestro-v3.0.0.csv
├── maestro-v3.0.0.json
├── 2004/
├── 2006/
└── ...
```

你当前给出的 Windows 数据目录可以直接作为 `--data-root`：

```powershell
$DataRoot = "E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0"
```

也就是说，这一层目录下面应该能看到 `maestro-v3.0.0.csv` 和年份子目录。

## 运行顺序：不用先单独造错

当前版本的推荐流程是：**先安装项目和依赖，然后直接运行训练脚本；不需要提前生成一个“错误 MIDI 数据集”文件夹**。训练脚本现在支持两阶段训练：先用全对数据训练，再把同一个数据集切换成在线造错后继续训练。原因是 `MaestroWrongNoteDataset.__getitem__` 每次取训练窗口时都会调用 `corrupt_note_window(...)`，当 `error_rate=0` 时就是全对数据，当 `error_rate>0` 时就是造错数据，训练样本会自动包含：

- `features`：造错后的输入 note 特征。
- `is_error`：每个输入 note 是否为错音/误触。
- `target_pitch`：这个输入 note 应该纠正到的正确 MIDI pitch。
- `mask`：padding mask。

所以你现在应该按下面顺序运行：

1. 进入 `code_new`。
2. 用你已有的 venv Python 安装项目。
3. 用 `--clean-epochs 1 --max-files 4 --epochs 1 --batch-size 2 --num-workers 0` 做最小测试：这会先训练 1 遍全对数据，再训练 1 遍造错数据。
4. 测试通过后，去掉 `--max-files`，保留 `--clean-epochs 1`，把 `epochs` 和 `batch-size` 调大开始正式训练。
5. 训练得到 `checkpoints\bigru_wrong_note.pt` 后，再用 `predict.py` 对新的 MIDI 做错音检测/纠错预测。

如果你只是想确认“造错”是否发生，可以先看训练日志：全对阶段会显示 `stage=clean`，造错阶段会显示 `stage=corrupt`。当前实现不会把造错后的 MIDI 另存到磁盘，而是在内存中动态生成训练样本。这样做的好处是同一首干净 MIDI 在不同 epoch/窗口中可以产生不同错误组合，数据增强效果更好。

## 训练

GPU 可用时脚本会自动使用 CUDA，否则使用 CPU。建议你先运行下面这个安装命令：

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new
E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe -m pip install -e .
```

然后先跑最小测试；确认没有路径、依赖或 CUDA 报错后，再做正式训练。通用训练命令如下：

```bash
python -m midi_error_detector.train \
  --data-root /path/to/maestro-v3.0.0 \
  --clean-epochs 1 \
  --epochs 20 \
  --batch-size 16 \
  --window-size 256 \
  --train-error-rate 0.15 \
  --error-rate 0.08 \
  --det-threshold 0.3 \
  --det-pos-weight 3.0 \
  --kind-class-weights 1 6 4 \
  --save-metric task_score \
  --lr-patience 4 \
  --lr-factor 0.5 \
  --lr-threshold 0.002 \
  --output checkpoints/bigru_wrong_note_theory_taskscore.pt
```

调试时建议先用少量文件跑通：

```bash
python -m midi_error_detector.train \
  --data-root /path/to/maestro-v3.0.0 \
  --clean-epochs 1 \
  --max-files 4 \
  --epochs 1 \
  --batch-size 2 \
  --train-error-rate 0.15 \
  --det-threshold 0.3
```

在你的 Windows 环境里，可以先跑这个最小 smoke test：

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new
E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe -m midi_error_detector.train `
  --data-root "E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0" `
  --clean-epochs 1 `
  --max-files 4 `
  --epochs 1 `
  --batch-size 2 `
  --train-error-rate 0.15 `
  --det-threshold 0.3 `
  --num-workers 0
```

确认能正常读取数据和使用 CUDA 后，再去掉 `--max-files` 并调大 `--batch-size`。Windows 下建议先设 `--num-workers 0`，避免多进程 DataLoader 与本地环境路径/编码问题混在一起；跑通后再逐步调高。正式训练示例：

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new
E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe -m midi_error_detector.train `
  --data-root "E:\downloads\桌面\dku\CS309\project\maestro-v3.0.0-midi\maestro-v3.0.0" `
  --clean-epochs 1 `
  --epochs 20 `
  --batch-size 16 `
  --window-size 256 `
  --train-error-rate 0.15 `
  --error-rate 0.08 `
  --det-threshold 0.3 `
  --det-pos-weight 3.0 `
  --kind-class-weights 1 6 4 `
  --save-metric task_score `
  --lr-patience 4 `
  --lr-factor 0.5 `
  --lr-threshold 0.002 `
  --num-workers 0 `
  --output checkpoints\bigru_wrong_note_theory_taskscore.pt
```

也可以使用仓库提供的 PowerShell 脚本，它已经把你给出的数据目录和 venv Python 作为默认值：

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new
.\scripts\train_windows.ps1 -CleanEpochs 1 -MaxFiles 4 -Epochs 1 -BatchSize 2 -TrainErrorRate 0.15 -DetThreshold 0.3
```

## 当前加入的乐理特征

为了让模型不只记 MIDI pitch 序列，现在 `note_features(...)` 的输入维度从 8 维扩展到 16 维。新增的乐理/上下文特征包括：

- **窗口级调性估计**：用当前 window 的 pitch-class duration histogram 和 Krumhansl major/minor profile 估计一个 major tonic 和 minor tonic。
- **音阶内外音提示**：每个 note 是否落在估计大调音阶、估计小调音阶内。
- **调性 profile 强度**：当前 pitch class 在估计 major/minor key profile 里的相对强度。
- **同 onset 和声上下文**：同一时间附近是否有其他音、和弦密度、最近垂直音程、是否存在协和音程。
- **旋律跳进大小**：当前 note 相对前一个 note 的绝对 pitch jump，用来帮助模型发现异常跳进。

这些特征仍然是轻量级的，不需要额外标注数据；它们只是给 BiGRU 一个“调性/和声/旋律”的 inductive bias。注意：因为输入维度变了，如果你想让新乐理特征真正参与训练，需要重新训练一个新 checkpoint，例如保存为 `checkpoints\bigru_wrong_note_theory_taskscore.pt`。旧 checkpoint 仍可用于推理兼容，但不会利用这些新增特征。

## 当前推荐的造错比例和检测阈值

根据前一次训练结果，模型 precision 高但 recall 低，说明它太保守、不愿意报错。现在默认做了两处调整：

- `--train-error-rate 0.15`：只提高训练阶段造错比例，让模型在训练时看到更多错音。
- `--error-rate 0.08`：held-out/test 仍保持较接近原来实验的造错比例，用来模拟更真实的应用错误率。
- `--det-threshold 0.3`：检测指标从 0.5 阈值降到 0.3，通常会提高 `det_recall`，但可能降低 `det_precision`；重点观察 `det_f1` 是否上升。
- `--det-pos-weight 3.0`：给“错音”正样本更高 BCE 权重，缓解 90% 以上都是正常音造成的类别不平衡。
- `--kind-class-weights 1 6 4`：给 `replace` 和 `delete` 动作更高权重，重点解决 `replace_kind_acc` 过低的问题。
- `--threshold-sweep 0.2 0.25 0.3 0.35 0.4 0.5`：每轮 held-out/test 验证会额外搜索这些阈值，并输出 `best_det_threshold`、`best_det_f1`、`best_det_precision`、`best_det_recall`。
- `--save-metric task_score`：checkpoint 不再只按 loss 保存，而是按 `0.50 * best_det_f1 + 0.25 * replace_pitch_top3 + 0.25 * replace_kind_acc` 保存，更贴近实际任务。
- `--lr-patience 4 --lr-factor 0.5 --lr-threshold 0.002`：如果 `task_score` 连续 4 个 corrupt epoch 没有至少 0.002 的实质提升，就把学习率减半，适合你现在这种 20 epoch 后提升变慢的情况。
- `--early-stop-patience 0`：默认不提前停止；如果你想自动停止，可以设成 `10` 或 `12`。

如果你想更激进地提升召回率，可以继续试 `--det-threshold 0.2` 或更高的 `--det-pos-weight`；如果误报太多，再调回 `0.4` 或降低 `--det-pos-weight`。

## 当前准确率还不能实用时，优先改什么

如果 held-out/test 上的指标仍然不能用于实际场景，不建议只盲目加 epoch。可以继续训练几轮，但更应该同时处理下面几个瓶颈：

1. **类别不平衡**：真实错误 note 只占少数，所以现在默认加入 `--det-pos-weight 3.0` 和 `--kind-class-weights 1 6 4`，让模型更重视错音、`replace` 和 `delete`。如果 `det_recall` 仍低，可以把 `--det-pos-weight` 试到 `4` 或 `5`；如果误报太多，降到 `2`。
2. **阈值选择**：实际使用时 precision/recall 的平衡主要由阈值控制。训练日志里的 `best_det_threshold` 来自 held-out/test 的 threshold sweep，比固定 0.3 更适合选择最终推理阈值。
3. **保存标准**：loss 下降不一定代表更能找错音，所以默认 `--save-metric task_score`，综合考虑检测 F1、top3 纠错和 replace 动作分类。
4. **乐理特征**：`note_features` 现在不只包含 pitch-class sin/cos，还加入窗口调性估计、大小调音阶内外音、Krumhansl key profile 强度、同 onset 和弦密度/最近和声音程/协和性、旋律跳进大小等特征，让 BiGRU 更容易学习“这个音是否像调外错音或不合理和声音”。
5. **数据量**：MAESTRO 已经比 1000 首小很多课程数据更可靠；如果准确率还不够，继续扩大到其他风格/难度/真实识谱错误 MIDI 会更有价值。单纯重复同一套合成错误会逐渐遇到上限。

经验上，下一轮可以先继续跑 `30` 或 `40` 个 corrupt epoch 观察是否还在上升；训练集现在会每个 epoch 重新采样造错，所以继续训练看到的错误组合会更多。若 `det_f1` 或 `task_score` 连续多轮不涨，再优先调 `det_pos_weight`、`kind_class_weights` 和 `train_error_rate`，而不是单纯加轮次。

## 为什么 20 epoch 后提升变慢

你这版 theory 模型在 10→20 epoch 还有明显提升，但 20→40 epoch 变慢，这不一定说明 BiGRU 选错了，也不一定只是样本量太少。更大的原因通常有三类：

1. **合成错误分布有限**：训练和 test 都是按规则造错，模型学到这些规则后会进入平台期。
2. **有效数据增强不足**：如果同一个 window 每轮看到的造错完全一样，继续训练就更像重复同一批样本。现在训练集会在每个 epoch 调整随机种子，重新采样造错；held-out/test 仍固定 seed，保证每轮指标可比。
3. **模型容量不是首要瓶颈**：当前 `replace_kind_acc` 和 `delete_kind_acc` 已经较高，说明 BiGRU 可以学到模式；真正限制更可能是合成错误和真实错误的分布差异、precision/recall 阈值选择、以及缺少真实 AI 识谱错误样本。

你这次最终结果是 `task_score=0.7426`、`best_det_f1=0.6451`、`replace_pitch_top3=0.8229`、`replace_kind_acc=0.8572`。这说明 theory 特征和重新造错是有效的，但 20→40 epoch 已经进入边际收益变小的阶段。现在训练脚本会在 `task_score` 连续几轮没有实质提升时自动降低学习率，避免后半程一直用同一个较大的 LR 小幅震荡。

所以建议不是马上换 Transformer 或继续盲目加大模型，而是先用当前 BiGRU + 每 epoch 重新造错 + plateau LR scheduler 再跑一版，对比 `task_score`、`best_det_f1`、`replace_pitch_top3` 是否超过上一版。如果仍然平台期，再考虑增加真实错误数据或更强模型。

## 正式训练时为什么一开始像是“没动”

正式训练去掉 `--max-files 4` 后，脚本会先解析完整 MAESTRO split 的 MIDI 文件并建立窗口索引；这一步发生在真正进入 epoch 训练之前，所以以前可能会看起来像卡住。现在训练脚本会明确打印：

```text
using device=cuda
building train loader: error_rate=0.0, max_files=None, cache_notes=True
loading train MIDI: ... file/s
train loader ready: files=..., windows=...
building test loader: error_rate=0.08, max_files=None, cache_notes=True
loading test MIDI: ... file/s
test loader ready: files=..., windows=...
clean 1/1: ... batch/s
corrupt train 1/20: ... batch/s
corrupt test 1/20: ... batch/s
```

如果你看到 `loading train MIDI` 或 `loading test MIDI`，说明程序正在解析 MIDI，不是停止了。完整数据集第一次加载可能会花几分钟，取决于硬盘速度和 CPU。进入 epoch 后，会看到 batch 级别的进度条。若你不想显示加载进度条，可以加 `--quiet`；但正式训练建议先不要加，方便确认程序在运行。

## 当前如何验证模型准度

训练脚本只在**造错后的 held-out split** 上验证；默认 `--eval-split test`，也就是在 `test` split 上按同样规则在线造错后评估。clean warm-up 阶段只用于训练，不会拿全对 test 算准度，也不会用全对 test 保存 checkpoint，因为你的应用场景是“输入 MIDI 里可能有错音，需要找出并纠正”。如果你想把 test 留到最终汇报，也可以改成 `--eval-split validation`。验证日志里的指标含义如下：

- `loss`：总 loss，等于错音检测 loss 加上加权后的 pitch 纠正 loss；它主要用来判断整体是否在下降。
- `det_loss`：错音检测二分类 loss。
- `det_threshold`：本次统计 precision/recall/F1 使用的检测阈值。
- `best_det_threshold` / `best_det_f1` / `best_det_precision` / `best_det_recall`：在 `--threshold-sweep` 给出的多个阈值里，held-out/test F1 最好的阈值及其指标。
- `task_score`：用于保存 checkpoint 的综合任务分数，默认综合 `best_det_f1`、`replace_pitch_top3` 和 `replace_kind_acc`。
- `pitch_loss`：正确 MIDI pitch 的 128 类分类 loss；删除型误触不参与这个 pitch loss。
- `kind_loss`：动作分类 loss，三类动作是 `keep`、`replace`、`delete`。
- `pitch_acc`：所有非删除 note 上预测正确 pitch 的比例；这个指标会被大量本来就正确的音影响。
- `replace_pitch_top1`：**只在需要替换的错音上**，top1 预测 pitch 是否等于正确 pitch。
- `replace_pitch_top3`：**只在需要替换的错音上**，top3 候选 pitch 是否包含正确 pitch。
- `kind_acc`：三分类动作整体准确率。
- `replace_kind_acc`：真实需要替换的错音里，有多少被分类成 `replace`。
- `delete_kind_acc`：真实需要删除的误触音里，有多少被分类成 `delete`。
- `det_acc`：错音检测整体准确率。
- `det_precision`：模型判为错音的 note 里，有多少确实是错音；precision 低说明误报多。
- `det_recall`：真实错音里，有多少被模型找出来；recall 低说明漏检多。
- `det_f1`：precision 和 recall 的综合指标，适合汇报错音检测效果。
- `error_rate`：当前 batch/window 里实际合成出来的错音比例，用来确认造错数据是否存在。
- `replace_rate`：需要替换 pitch 的错音比例。
- `delete_rate`：需要删除的额外误触音比例。

所以正式实验时建议重点看 corrupt 阶段 held-out/test 指标：`det_f1`、`det_precision`、`det_recall` 看“有没有找出错音”；`kind_acc`、`replace_kind_acc`、`delete_kind_acc` 看“分类成替换还是删除是否正确”；`replace_pitch_top1` 和 `replace_pitch_top3` 看“弹错类错误能不能给出正确音，top3 是否包含正确音”。`pitch_acc` 可以保留作为辅助指标，但它不是这个任务最核心的指标。

## 如何解读 smoke test 结果

如果你的小训练集输出类似：

```text
stage=clean epoch=1/1 train={'loss': 1.98, 'det_loss': 0.006, 'pitch_loss': 3.95, 'pitch_acc': 0.038} valid={'loss': 1.76, 'det_loss': 0.001, 'pitch_loss': 3.52, 'pitch_acc': 0.052}
saved checkpoints\bigru_wrong_note.pt
stage=corrupt epoch=1/1 train={'loss': 1.90, 'det_loss': 0.348, 'pitch_loss': 3.11, 'pitch_acc': 0.090} valid={'loss': 1.85, 'det_loss': 0.308, 'pitch_loss': 3.08, 'pitch_acc': 0.073}
```

这说明代码流程是正常的：

- `stage=clean` 已经完成，表示模型先在全对数据上训练了一遍；这一阶段只看 train 日志，不对全对 test 计算准度。
- `stage=corrupt` 已经完成，表示模型又在在线造错数据上继续训练了一遍，并且在造错后的 held-out/test split 上验证。
- `saved checkpoints\bigru_wrong_note.pt` 表示基于造错 held-out/test 指标保存了 checkpoint，不是只安装了包。
- `det_loss` 在 clean 阶段很低是正常的，因为全对阶段几乎所有 `is_error` 都是 0。
- `det_loss` 在 corrupt 阶段变高也是正常的，因为造错阶段开始出现正样本，检测任务变难。
- `pitch_acc`、`replace_pitch_top1` 或 `replace_pitch_top3` 在 smoke test 里偏低也正常：你只用了 `--max-files 4`、`--epochs 1`、`--batch-size 2`，这只是检查代码能不能跑通，不是最终效果。

下一步建议：

1. 保留 `--clean-epochs 1`。
2. 去掉 `--max-files 4`，使用完整 MAESTRO train split。
3. 先用 `--epochs 10` 或 `--epochs 20` 正式训练。
4. 如果 GPU 显存足够，把 `--batch-size` 从 2 提到 8、16 或 32。
5. 正式训练时重点看 held-out/test 的 `det_f1`、`det_precision`、`det_recall`、`kind_acc`、`replace_pitch_top1`、`replace_pitch_top3`，同时确认 `loss` 和 `det_loss` 是否整体下降。

## 40 epoch weighted 训练结果解读

你这次 `checkpoints\bigru_wrong_note_weighted_taskscore.pt` 的结果已经比前几版明显更接近可用，但还不能说是最终产品级模型：

- `best_det_f1=0.6397`，比固定 `det_threshold=0.3` 下的 `det_f1=0.5663` 更好，说明实际推理应该优先使用 checkpoint 里保存的 `best_det_threshold=0.5`。
- `best_det_precision=0.5375`、`best_det_recall=0.7901`：能找出大多数合成错音，但误报仍然偏多；如果你的应用更怕误报，可以手动用 `--threshold 0.6` 做更保守推理。
- `replace_pitch_top1=0.4243`、`replace_pitch_top3=0.8024`：top3 候选已经比较有价值，实际界面/后处理建议展示 top3，而不是只相信 top1。
- `replace_kind_acc=0.8192`、`delete_kind_acc=0.8963`：加权 loss 后动作分类已经明显改善，说明 `replace/delete` 少数类权重是有效的。
- `task_score=0.7252`：这是当前默认保存 checkpoint 的综合指标，后续实验可以用它来比较模型。

下一步如果还想提升实际可用性，优先级建议是：

1. 由于现在加入了 16 维乐理特征、每 epoch 重新采样造错、以及 plateau 后自动降低学习率，建议重新训练一版 `checkpoints\bigru_wrong_note_theory_taskscore.pt`，可以先跑 `40` 个 corrupt epoch；如果 20→40 仍然只小幅提升，就不用盲目拉到 80。
2. 在真实应用中用 checkpoint 默认阈值 `0.5` 起步；如果误报多，试 `--threshold 0.6`；如果漏检多，试 `--threshold 0.4`。
3. 不要只增加合成数据量；更有价值的是收集少量真实 AI 识谱错误 MIDI，作为额外 held-out test 或微调数据。
4. 如果这版 theory baseline 仍不够，可以继续加入更细的节拍强弱、局部和弦标签、经过音/辅助音识别；但每次只加一类特征，方便对比 `task_score` 是否真的提升。

## theory 40 epoch 最终结果解读

你这次 theory 版本最终 test 指标是：`task_score=0.7426`、`best_det_f1=0.6451`、`best_det_precision=0.5287`、`best_det_recall=0.8271`、`replace_pitch_top1=0.4455`、`replace_pitch_top3=0.8229`、`replace_kind_acc=0.8572`、`delete_kind_acc=0.9391`。

结论：这版比旧 weighted baseline 略有提升，尤其 top3、replace/delete 分类更好；但 precision 仍在 0.53 左右，说明它仍更适合作为“错音候选提示 + top3 建议”，不适合完全自动改谱。20 epoch 后继续提升很慢时，优先尝试 scheduler、真实错误数据和后处理阈值，而不是只加 epoch。

## 推理

训练结束后会保存 checkpoint，例如 `checkpoints\bigru_wrong_note.pt`。之后可以对单个 MIDI 做推理：

```bash
python -m midi_error_detector.predict \
  --checkpoint checkpoints/bigru_wrong_note.pt \
  --midi /path/to/input.mid \
  --top-k 3
```

你的 Windows 路径写法示例：

```powershell
cd E:\downloads\桌面\dku\CS309\project\code_new
E:\downloads\桌面\dku\CS309\project\code\venv\Scripts\python.exe -m midi_error_detector.predict `
  --checkpoint checkpoints\bigru_wrong_note_weighted_taskscore.pt `
  --midi "E:\path\to\your_input.mid" `
  --top-k 3
```

输出 JSON 中包含可疑 note 的位置、原 pitch、预测动作 `keep`/`replace`/`delete`、动作概率、top-k 候选正确 pitch、错误概率与实际使用的 `detection_threshold`。如果你不手动传 `--threshold`，`predict.py` 会优先读取 checkpoint 里的 `valid_metrics.best_det_threshold`；你这次 40 epoch 结果里 test 的最佳阈值是 `0.5`，所以实际推理会默认比训练日志里的 `det_threshold=0.3` 更保守。对于 `replace`，看 `top_pitches`；对于 `delete`，表示这个 note 更可能是多余误触，后处理时应删除而不是替换成另一个 pitch。

## 后续可改进方向

- 加入和弦上下文特征，例如同 onset 里的音程集合。
- 区分“错音需替换”和“多余误触需删除”，为误触音新增 delete/no-op 动作头。
- 用节拍网格或 quantization 后的位置特征替代简单 onset 小数周期。
- 对不同错误类型设置不同采样比例，贴近你的实际识谱错误分布。

## 实现备注

`MaestroWrongNoteDataset` 默认会在初始化时解析每个 MIDI 文件，并把 note event 缓存在内存里；这样训练窗口不会反复解析同一个 MIDI 文件，速度会比每个样本都读盘快很多。如果内存不足，可以给训练脚本加 `--no-cache-notes`。
