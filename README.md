股票价格预测项目

项目简介

本项目旨在通过机器学习方法，对股票的次日收盘价进行预测，这里以纳斯达克100指数为例。项目涵盖了从原始数据清洗、特征工程、模型训练与评估到最佳模型保存的完整流程。目前，项目主要使用线性回归模型、MLP非线性回归模型进行价格预测，并使用Transformer模型进行涨跌方向的二分类预测，全面评估了它们在不同任务上的表现。

项目功能

数据预处理： 清洗原始股票数据，处理缺失值、特殊字符和数据类型转换，统一数据格式。

特征工程： 基于历史收盘价和日期，生成多种技术分析指标和时间特征，包括滞后价格、移动平均线、波动率、RSI、MACD以及周、月、季度信息。

目标变量分离： 将预测目标（次日收盘价）从特征数据中独立分离并保存，避免数据泄露。

模型训练与评估： 训练并评估线性回归（包括线性回归、岭回归和Lasso回归）、多层感知机（MLP）回归模型，以及基于Transformer的二分类模型。使用MAE、RMSE、R²和涨跌方向准确率等指标进行回归模型性能衡量；使用准确率、精确率、召回率、F1分数、ROC AUC等指标进行分类模型性能衡量。

特征相关性分析： 对特征工程后的数据进行相关性分析，并可视化为热力图，帮助理解特征间的关系。

最佳模型保存： 自动选择表现最佳的模型，并在完整数据集上重新训练后，将其与特征缩放器一同保存，以便后续部署和预测。

项目结构

.
├── data/

│   ├── 纳斯达克100指数历史数据.csv  # 原始数据文件 (需要用户自行提供)

│   ├── 纳斯达克100指数_processed.csv # 预处理后的数据 (由data_preprocessing.py生成)

│   ├── 纳斯达克100_index_with_features.csv # 包含特征但不含目标变量的数据 (由feature_engineering.py生成)

│   └── 纳斯达克100_target_next_day_close.csv # 仅包含日期和目标变量的数据 (由feature_engineering.py生成)

├── data_preprocessing.py          # 数据预处理脚本

├── feature_engineering.py          # 特征工程脚本

├── train_xgboost.py                # 线性模型训练、评估与保存脚本

├── train_mlp.py                     # MLP回归模型训练、评估与保存脚本

├── train_transformer_classifier.py  # Transformer分类模型训练、评估与保存脚本

├── feature_correlation_analysis.py  # 特征相关性分析脚本

├── README.md                      # 项目说明文件

└── requirements.txt                # 项目依赖库文件 (需要手动创建)

环境搭建

克隆仓库：

git clone <你的GitHub仓库URL>
cd <你的项目文件夹>

创建并激活虚拟环境 (推荐)：

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

安装依赖：

请手动创建一个 requirements.txt 文件，内容如下：

pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
tensorflow
flask
flask-cors

然后运行：

pip install -r requirements.txt

准备原始数据：

在 data/ 目录下放置你的原始纳斯达克100指数历史数据CSV文件，并确保其命名为 纳斯达克100指数历史数据.csv。该文件应包含“日期”、“收盘”、“开盘”、“高”、“低”、“交易量”、“涨跌幅”等列。

使用方法

按照以下顺序运行脚本，完成数据处理、特征工程、模型训练和相关性分析：

运行数据预处理：

python data_preprocessing.py

此脚本将读取 纳斯达克100指数历史数据.csv，进行清洗和预处理，并生成 纳斯达克100指数_processed.csv。

运行特征工程：

python feature_engineering.py

此脚本将读取 纳斯达克100指数_processed.csv，生成各种预测特征，并将特征和目标变量分别保存到 纳斯达克100_index_with_features.csv 和 纳斯达克100_target_next_day_close.csv。

运行特征相关性分析：

python feature_correlation_analysis.py

此脚本将读取 纳斯达克100_index_with_features.csv，计算特征相关性矩阵，并生成热力图保存到 result_figure/ 目录下。

运行模型训练与评估（线性模型）：

python train_xgboost.py

此脚本将加载特征和目标数据，划分训练集和测试集，训练并评估线性回归模型，打印详细的评估结果（包括MAE、RMSE、R²和涨跌方向准确率），并生成模型性能对比图。最后，它会将表现最佳的线性模型和特征缩放器保存到项目根目录。

运行模型训练与评估（MLP回归模型）：

python train_mlp.py

此脚本将加载特征和目标数据，划分训练集和测试集，训练并评估多层感知机（MLP）回归模型，打印详细的评估结果，并生成MLP预测结果图。最后，它会将表现最佳的MLP模型和特征缩放器保存到项目根目录。

运行模型训练与评估（Transformer分类模型）：

python train_transformer_classifier.py

此脚本将加载特征和目标数据，将其转换为序列数据，划分训练集和测试集，训练并评估基于Transformer的二分类模型，打印详细的分类评估结果。最后，它会将表现最佳的Transformer模型和特征缩放器保存到项目根目录。

未来改进方向

涨跌方向预测优化：

将问题转化为分类问题（预测“涨”或“跌”），并使用分类模型（如逻辑回归、决策树、随机森林分类器、XGBoost分类器等）进行训练，直接优化方向准确率。

探索多任务学习，同时预测价格和方向。

更丰富的特征工程：

引入更多技术分析指标（如布林带、随机指标、ATR等）。

探索特征之间的交互项。

考虑整合外部数据，如宏观经济指标、新闻情绪、公司基本面数据等。

进行特征选择或降维，优化特征集。

模型多样化与超参数调优：

重新引入并优化更复杂的非线性模型（如随机森林回归、XGBoost回归、LightGBM回归，甚至LSTM），在确保特征缩放正确的前提下，进行系统的超参数调优。

采用时间序列交叉验证等更稳健的评估方法。

针对 Transformer 分类模型的优化：

类别不平衡处理： 尽管已使用 class_weight，但如果模型仍倾向于预测多数类，可能需要更激进的策略，例如过采样 (Oversampling) 或欠采样 (Undersampling)，或调整分类阈值。

模型结构调整： 尝试增加 Transformer Block 的数量、调整 embed_dim、num_heads、ff_dim、mlp_units 和 dropout_rate，以增强模型学习复杂模式的能力。

损失函数： 探索 Focal Loss 等专门用于处理类别不平衡的损失函数。

模型可解释性：

使用SHAP或LIME等工具，解释模型预测的驱动因素，增强模型透明度。

部署与自动化：

构建一个完整的自动化预测管道，包括数据自动获取、处理、预测和结果输出。

开发一个简单的预测界面，方便用户进行交互式预测和结果可视化。

接入实时股票数据API，实现实时预测。

贡献

欢迎对本项目提出建议或贡献代码！如果您有任何问题或改进意见，请随时通过GitHub Issues提出。
