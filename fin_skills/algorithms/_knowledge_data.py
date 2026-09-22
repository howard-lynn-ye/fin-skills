"""Curated method references, not execution registrations or performance claims.

Rows: stable ID | English label | Chinese label | distinguishing constraint.
Sources establish that a method exists; requirements/cautions are our editorial notes.
Do not copy upstream code or describe a reference as an installed adapter.
"""

VERIFIED_ON = "2026-09-21"
SOURCES = {
    "french": ("Kenneth French data library", "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"),
    "qc": ("QuantConnect strategy examples", "https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library"),
    "options": ("LEAN option strategies", "https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/option-strategies"),
    "carry": ("AQR: Carry", "https://www.aqr.com/Insights/Research/Journal-Article/Carry"),
    "trend": ("AQR: Time Series Momentum", "https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum"),
    "value": ("AQR: Value and Momentum Everywhere", "https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere"),
    "hummingbot": ("Hummingbot strategies", "https://hummingbot.org/strategies/"),
    "avellaneda": ("Hummingbot Avellaneda implementation", "https://hummingbot.org/strategies/v1-strategies/avellaneda-market-making/"),
    "amm": ("Hummingbot AMM arbitrage", "https://hummingbot.org/strategies/v1-strategies/amm-arbitrage/"),
    "execution": ("LEAN execution models", "https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/execution/supported-models"),
    "controls": ("LEAN risk controls", "https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/risk-management/supported-models"),
    "portfolio": ("Riskfolio portfolio objectives", "https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html"),
    "frontier": ("PyPortfolioOpt efficient frontiers", "https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html"),
    "ta": ("TA-Lib function catalog", "https://ta-lib.org/functions/"),
    "tsa": ("statsmodels time-series methods", "https://www.statsmodels.org/stable/tsa.html"),
    "supervised": ("scikit-learn supervised estimators", "https://scikit-learn.org/stable/supervised_learning.html"),
    "unsupervised": ("scikit-learn unsupervised estimators", "https://scikit-learn.org/stable/unsupervised_learning.html"),
    "rl": ("Stable Baselines3 algorithm support", "https://stable-baselines3.readthedocs.io/en/master/guide/algos.html"),
    "neural": ("NeuralForecast model catalog", "https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html"),
    "comparison": ("arch multiple comparison methods", "https://arch.readthedocs.io/en/latest/multiple-comparison/multiple-comparison_examples.html"),
    "cv": ("scikit-learn cross validation", "https://scikit-learn.org/stable/modules/cross_validation.html"),
    "cvx": ("Cvxportfolio trading policies", "https://www.cvxportfolio.com/en/stable/policies.html"),
    "olmar": ("Li and Hoi: Online Moving Average Reversion", "https://arxiv.org/abs/1206.4626"),
    "corporate": ("AQR corporate arbitrage description", "https://ucits.aqr.com/Insights/Fund-Promo/AQR-Corporate-Arbitrage"),
    "quality": ("AQR: Quality Minus Junk", "https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk"),
    "basis": ("Hummingbot spot-perpetual strategy", "https://hummingbot.org/strategies/v1-strategies/spot-perpetual-arbitrage/"),
    "funding": ("Hummingbot funding-rate strategy source", "https://github.com/hummingbot/hummingbot/blob/master/scripts/v2_funding_rate_arb.py"),
}

GROUPS = []


def group(kind, family, assets, inputs, source, status, rows):
    GROUPS.append(dict(kind=kind, family=family, asset_classes=assets.split(),
                       required_data=inputs.split(), source_id=source, status=status,
                       rows=rows.strip().splitlines()))


group("strategy", "factor", "equity", "point_in_time_fundamentals universe prices", "french", "reference", """
value_factor|Value factor|价值因子|财报披露日滞后和退市样本必须保留。
size_factor|Size factor|规模因子|小盘成交容量和停牌不能忽略。
profitability_factor|Profitability factor|盈利能力因子|不可使用事后重述的财务数据。
investment_factor|Investment factor|投资因子|资产增长需按当时可得财报计算。
dividend_yield_factor|Dividend yield factor|股息率因子|除息调整不等于当时知道未来分红。
earnings_price_factor|Earnings yield factor|盈利收益率因子|负盈利样本的处理需预先规定。
cashflow_price_factor|Cash-flow yield factor|现金流因子|会计口径和发布时间需一致。
reversal_factor|Short-term reversal|短期反转因子|换手成本可能吞掉毛收益。
""")
group("strategy", "trend", "equity futures fx commodity", "prices contract_rolls", "trend", "reference", """
time_series_trend|Time-series trend|时间序列趋势|交易在信号可得之后；期货连续价格不可直接当成交价。
""")
group("strategy", "factor", "equity futures fx", "prices valuation_data universe", "value", "reference", """
cross_asset_value_momentum|Value and momentum combination|跨资产价值动量|资产间估值定义不同；先固定定义再评估组合。
""")
group("strategy", "carry", "fx fixed_income futures commodity", "spot forward_rates funding contract_terms", "carry", "reference", """
fx_carry|Currency carry|外汇套息|利差不是总回报；需计入汇率变动和融资。
bond_carry|Bond carry|债券持有收益|久期变化和利率冲击会改变持有期收益。
commodity_carry|Commodity carry|商品期限结构收益|展期规则与实际可交易合约必须明确。
""")
group("strategy", "cross_section", "equity etf", "prices point_in_time_universe", "qc", "external", """
residual_momentum|Residual momentum|残差动量|因子回归只能用历史窗口估计。
sector_rotation|Sector rotation|行业轮动|行业分类和指数成分要使用历史版本。
country_momentum|Country momentum|国家指数动量|时区、汇率和交易日需要对齐。
low_volatility_selection|Low-volatility selection|低波动选股|低历史波动不代表未来低风险。
liquidity_factor|Liquidity selection|流动性选股|不能假设低流动性资产按收盘价全部成交。
accrual_factor|Accrual selection|应计项选股|会计数据必须按披露时点对齐。
reit_momentum|REIT momentum|房地产信托动量|需保留分红和退市回报。
""")
group("strategy", "relative_value", "equity etf commodity fx", "aligned_prices borrow_costs", "qc", "external", """
distance_pairs|Distance pairs|距离配对交易|训练期选对；测试期不能重新用全样本挑选。
cointegration_pairs|Cointegration pairs|协整配对交易|协整检验不等于完整交易策略。
copula_pairs|Copula pairs|联结函数配对交易|尾部依赖和参数漂移需要检验。
oil_spread|WTI-Brent spread|原油跨品种价差|合约到期和运输约束会造成结构性价差。
""")
group("strategy", "timing", "equity etf futures fx", "ohlcv calendar", "qc", "external", """
dual_thrust|Dual Thrust|双重推力突破|当日完整高低价不能进入当日早先决策。
dynamic_breakout|Dynamic breakout|动态通道突破|波动窗口和突破阈值只使用历史数据。
overnight_effect|Overnight effect|隔夜效应|收盘竞价是否可成交需要明确。
turn_of_month|Turn-of-month effect|月末月初效应|节假日和交易成本会改变持有区间。
expiry_week|Expiry-week effect|期权到期周效应|交易日历不能用固定自然日替代。
market_filtered_momentum|Market-filtered momentum|市场状态过滤动量|不能用事后牛熊标签过滤历史交易。
""")
group("strategy", "options", "options", "option_chain quotes contract_terms underlying_prices", "options", "external", """
covered_call|Covered call|备兑看涨|保留下跌风险并限制上涨收益。
protective_put|Protective put|保护性看跌|权利金和到期后的保护空档需计入。
collar|Collar|领口组合|多腿成交价格与行权风险需共同建模。
bull_call_spread|Bull call spread|牛市看涨价差|必须匹配到期日、合约乘数和行权方式。
bear_put_spread|Bear put spread|熊市看跌价差|多腿不能都假设按有利报价成交。
long_straddle|Long straddle|买入跨式|方向中性不等于对波动率和时间价值中性。
short_straddle|Short straddle|卖出跨式|尾部损失与保证金追加需单独评估。
long_strangle|Long strangle|买入宽跨式|需要足够大的价格变动覆盖双腿权利金。
iron_condor|Iron condor|铁鹰组合|到期和提前行权可能改变名义风险边界。
butterfly|Butterfly spread|蝶式价差|中间执行价附近收益对成交误差敏感。
calendar_spread|Calendar spread|日历价差|不同到期日的隐含波动不能当成同一参数。
box_spread|Box spread|盒式价差|融资、行权方式和手续费决定实际收益。
""")
group("strategy", "market_making", "crypto", "order_book balances fees", "hummingbot", "external", """
pure_market_making|Pure market making|单市场做市|挂单排队和逆向选择不能用中间价成交替代。
cross_exchange_market_making|Cross-exchange market making|跨市场做市|对冲延迟和不同交易所余额约束需模拟。
""")
group("strategy", "market_making", "crypto", "order_book fills inventory", "avellaneda", "external", """
avellaneda_stoikov|Avellaneda-Stoikov quoting|库存约束做市|到达强度与风险参数需在历史训练段校准。
""")
group("strategy", "arbitrage", "crypto", "pool_state exchange_quotes fees gas", "amm", "external", """
amm_arbitrage|AMM arbitrage|自动做市池套利|池子滑点、燃气费与交易失败必须计入。
""")
group("decision", "execution", "equity etf futures fx", "target_positions quotes volume", "execution", "external", """
immediate_execution|Immediate execution|立即执行|目标仓位并不保证即时足量成交。
spread_execution|Spread-gated execution|价差阈值执行|等待价差收窄可能导致长期不成交。
std_execution|Deviation-gated execution|偏离阈值执行|价格偏离门槛可能改变信号原本的持有期。
vwap_execution|VWAP execution model|成交量加权执行模型|盘中可得数据与事后全日成交量需区分。
""")
group("decision", "risk_control", "equity etf", "positions nav sector_map quotes", "controls", "external", """
position_drawdown_stop|Position drawdown stop|单仓回撤止损|跳空会穿越止损阈值。
portfolio_drawdown_stop|Portfolio drawdown stop|组合回撤止损|平仓后再次入场规则需要显式定义。
trailing_stop|Trailing stop|移动止损|只使用已观测高水位且成交在触发之后。
profit_take|Profit-taking rule|止盈规则|提前止盈会截断收益分布上尾。
sector_exposure_cap|Sector exposure cap|行业敞口上限|行业归属和绝对敞口定义影响约束结果。
""")
group("decision", "allocation", "multi_asset", "asset_returns constraints", "portfolio", "external", """
mean_mad|Mean absolute deviation allocation|平均绝对偏差配置|样本依赖和换手成本需检验。
mean_semivariance|Semivariance allocation|半方差配置|下行阈值必须与投资目标一致。
mean_cvar|CVaR allocation|条件风险价值配置|尾部样本少时最优权重不稳定。
mean_evar|EVaR allocation|熵风险价值配置|求解器支持与数值尺度需要检查。
mean_cdar|CDaR allocation|条件回撤配置|回撤依赖样本路径顺序。
mean_edar|EDaR allocation|熵回撤配置|样本路径与尾部概率选择影响解。
mean_ulcer|Ulcer allocation|溃疡指数配置|路径风险不是独立单期损失的分布。
worst_case_allocation|Worst-case allocation|最坏情形配置|不确定集选择本身需要验证。
risk_budget_allocation|Risk budgeting|风险预算配置|风险贡献依赖风险模型和约束。
factor_risk_budget|Factor risk budgeting|因子风险预算|因子相关性和遗漏因子会影响归因。
owa_allocation|Ordered weighted averaging|有序加权配置|排序损失的权重必须预先确定。
""")
group("decision", "allocation", "multi_asset", "asset_returns constraints", "frontier", "external", """
semivariance_frontier|Semivariance frontier|半方差前沿|目标收益率可能不可行。
cvar_frontier|CVaR frontier|条件风险价值前沿|置信水平越高所需尾部样本越多。
cdar_frontier|CDaR frontier|条件回撤前沿|收益矩阵需要保持时间顺序。
""")
group("algorithm", "technical", "multi_asset", "ohlcv", "ta", "external", """
ta_adx|ADX|平均趋向指标|指标本身没有规定交易方向。
ta_aroon|AROON|阿隆指标|窗口长度会改变极值年龄。
ta_cci|CCI|商品通道指标|阈值不是跨资产统一尺度。
ta_mfi|MFI|资金流量指标|成交量口径应保持一致。
ta_stoch|STOCH|随机震荡指标|平滑参数影响信号延迟。
ta_willr|WILLR|威廉指标|相同窗口极值不表示反转已发生。
ta_atr|ATR|平均真实波幅|价格单位量不能直接跨资产比较。
ta_natr|NATR|归一化真实波幅|归一化不消除跳空风险。
ta_obv|OBV|能量潮|拆股和成交量调整必须一致。
ta_ad|AD|累积派发指标|日线位置不能解释日内实际资金流。
ta_adosc|ADOSC|累积派发振荡|不同平滑长度产生不同滞后。
ta_kama|KAMA|自适应均线|自适应平滑仍需历史预热。
ta_sar|SAR|抛物线转向|趋势反复时可能频繁换手。
ta_trix|TRIX|三重平滑变化率|多次平滑增加响应延迟。
ta_hilbert|HT_TRENDMODE|希尔伯特趋势模式|边界和预热行为应与实际版本核对。
""")
group("algorithm", "time_series", "multi_asset", "ordered_series", "tsa", "external", """
sarimax|SARIMAX|季节自回归外生模型|未来外生变量必须是真正已知或独立预测。
var|VAR|向量自回归|参数随变量数量快速增长。
vecm|VECM|向量误差修正|协整秩应仅在训练段确定。
unobserved_components|Unobserved components|不可观测成分模型|滤波输出与全样本平滑输出不可混用。
dynamic_factor|Dynamic factor|动态因子模型|因子符号与尺度存在识别约定。
markov_regression|Markov regression|马尔可夫转换回归|平滑状态包含未来信息。
markov_autoregression|Markov autoregression|马尔可夫转换自回归|转换概率需在训练段估计。
exponential_smoothing|Exponential smoothing|指数平滑|趋势外推不能忽略结构性断点。
stl_decomposition|STL|季节趋势分解|双向分解不能直接生成历史实时信号。
adf_test|ADF|增广单位根检验|拒绝单位根不证明策略盈利。
kpss_test|KPSS|平稳性检验|原假设与ADF不同。
granger_test|Granger test|格兰杰预测检验|预测先后关系不能当作因果干预证据。
""")
group("algorithm", "supervised", "multi_asset", "features labels", "supervised", "external", """
elastic_net|Elastic net|弹性网|标准化只能在训练段拟合。
svr|SVR|支持向量回归|核参数和尺度影响预测。
svc|SVC|支持向量分类|类别置信分数不等于校准概率。
knn|Nearest neighbors|近邻学习|距离对特征尺度敏感。
extra_trees|Extra trees|极端随机树|随机化不能消除标签泄漏。
hist_gradient_boosting|Histogram boosting|直方图梯度提升|早停验证必须遵守时间顺序。
gaussian_process|Gaussian process|高斯过程|核假设和训练规模限制适用性。
naive_bayes|Naive Bayes|朴素贝叶斯|条件独立假设在金融特征中常不成立。
mlp|MLP|多层感知机|验证集不能参与缩放器拟合。
isotonic_calibration|Isotonic calibration|保序概率校准|校准数据应独立于拟合数据。
""")
group("algorithm", "regime", "multi_asset", "features", "unsupervised", "external", """
kmeans_regime|K-means clustering|聚类状态识别|簇标签不是事先存在的牛熊标签。
gmm_regime|Gaussian mixtures|高斯混合状态|全样本聚类会泄漏未来分布。
dbscan_regime|DBSCAN|密度聚类|噪声点不应自动变成交易信号。
spectral_regime|Spectral clustering|谱聚类|历史全样本图不能作为在线决策输入。
ica_factors|ICA|独立成分因子|成分排序和符号在重拟合时可能变化。
nmf_factors|NMF|非负矩阵分解|有符号收益不能直接作为非负输入。
""")
group("algorithm", "reinforcement_learning", "multi_asset", "gymnasium_environment", "rl", "external", """
a2c|A2C|优势行动者评论家|环境奖励和成本建模决定所学行为。
ddpg|DDPG|深度确定性策略梯度|连续动作需映射为可行仓位。
dqn|DQN|深度Q学习|离散动作集合需由调用方明确。
td3|TD3|双延迟策略梯度|训练环境不能包含未来市场状态。
""")
group("algorithm", "forecast", "multi_asset", "unique_id ds y", "neural", "external", """
nbeats|N-BEATS|基展开预测网络|窗口切分必须先于训练样本构造。
tft|TFT|时序融合Transformer|已知未来变量和未知变量需分别声明。
deepar|DeepAR|概率自回归网络|分布预测需检查校准。
itransformer|iTransformer|倒置Transformer|多变量对齐不能使用事后填补。
timemixer|TimeMixer|时序混合网络|尺度分解需遵守训练测试边界。
""")
group("decision", "evaluation", "multi_asset", "strategy_losses benchmark_losses", "comparison", "external", """
spa_selection|SPA comparison|优越预测能力检验|必须记录实际尝试过的候选全集。
stepm_selection|StepM comparison|逐步多重比较|重复筛选仍会改变统计含义。
model_confidence_set|Model confidence set|模型置信集合|保留候选集合不等于识别唯一最佳策略。
""")
group("decision", "evaluation", "multi_asset", "features labels timestamps", "cv", "external", """
temporal_cv|Time-series split|时间顺序验证|按时间切分仍需处理重叠标签泄漏。
grouped_cv|Grouped split|分组验证|组隔离不能替代时间隔离。
nested_cv|Nested validation|嵌套验证|内外层均应采用符合金融时间结构的切分。
""")

group("strategy", "event_driven", "equity convertible", "announcements prices borrow_costs contract_terms", "corporate", "reference", """
merger_arbitrage|Merger arbitrage|并购套利|交易失败、要约条款与完成时间均存在不确定性。
convertible_arbitrage|Convertible arbitrage|可转债套利|股票对冲不能消除信用、流动性和赎回风险。
""")
group("strategy", "factor", "equity", "point_in_time_fundamentals prices universe", "quality", "reference", """
quality_minus_junk|Quality minus junk|质量因子多空|质量定义和做空约束必须事先确定。
""")
group("strategy", "arbitrage", "crypto", "spot_prices perpetual_prices funding margin", "basis", "external", """
spot_perpetual_basis|Spot-perpetual basis|现货永续基差|价差收敛前可能触发保证金约束。
""")
group("strategy", "arbitrage", "crypto", "funding_rates quotes margin fees", "funding", "external", """
funding_rate_arbitrage|Funding-rate arbitrage|资金费率套利|当前费率不保证未来费率或两腿同时成交。
""")
group("decision", "online_allocation", "multi_asset", "historical_price_relatives", "olmar", "reference", """
olmar|OLMAR|在线均值回归配置|价格相对值的时间顺序与交易成本决定可交易性。
""")
group("decision", "allocation", "multi_asset", "asset_returns forecasts costs constraints", "cvx", "external", """
single_period_optimization|Single-period optimization|单期成本约束优化|预测风险、交易成本和持仓成本需使用相同周期。
multi_period_optimization|Multi-period optimization|多期成本约束优化|未来路径是预测输入，不能使用真实未来收益。
periodic_rebalance|Periodic rebalance|定期再平衡|日历和目标仓位要在交易前确定。
proportional_rebalance|Proportional rebalance|比例再平衡|逐步接近目标会改变风险暴露时间。
adaptive_rebalance|Adaptive rebalance|偏离阈值再平衡|触发阈值应在独立数据上验证。
hold_policy|Hold policy|保持仓位决策|不交易仍有市场和融资风险。
uniform_allocation|Uniform allocation|等权配置决策|等资本权重不等于等风险。
""")

# Related components are not full implementations of the curated strategy.
RELATED_MODELS = {
    "time_series_trend": ["momentum", "vol_target_momentum"],
    "cointegration_pairs": ["engle_granger"],
    "market_filtered_momentum": ["rolling_market_state", "momentum"],
    "vwap_execution": ["vwap"],
    "covered_call": ["black_scholes"],
    "long_straddle": ["black_scholes", "quantlib_heston"],
}

ASSET_OVERRIDES = {
    "fx_carry": ["fx"], "bond_carry": ["fixed_income"],
    "commodity_carry": ["commodity", "futures"],
    "merger_arbitrage": ["equity"], "convertible_arbitrage": ["convertible", "equity"],
    "oil_spread": ["commodity", "futures"],
}

# Explicit gaps keep coverage auditable. This is a living inventory, not a claim of all methods.
GAPS = [
    "Event-driven: earnings drift, buybacks, index reconstitution; runnable merger policies",
    "Fixed income: curve trades, swap spreads and credit arbitrage; runnable convertible policies",
    "Macro: release surprises, nowcasting and point-in-time vintage policies",
    "Microstructure: queue models, optimal liquidation, limit-order-book predictors",
    "Decision learning: contextual bandits, further online allocation methods, Bayesian sizing",
    "Alternative data: news-event policies, supply chains, satellite and text signals",
    "Crypto: staking and liquidation policies; integrated funding and basis execution",
    "Evaluation: purged combinatorial validation and deflated performance statistics",
]
