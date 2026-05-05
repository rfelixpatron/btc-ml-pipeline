# Future Improvements & Roadmap

This document outlines potential enhancements to the BTC ML Predictive Pipeline to improve precision, reduce false positives, and move toward a production-ready trading system.

## 1. Feature Engineering Enhancements
- **On-Chain Data**: Integrate metrics like Exchange Net Flow, MVRV Ratio, and Hash Rate from providers like Glassnode or CryptoQuant.
- **Order Book Imbalance**: Add features derived from real-time L2 order book data (bid/ask pressure) to catch short-term liquidity shifts.
- **Alternative Sentiment**: Scrape Reddit (r/CryptoCurrency) or Twitter using specific cashtags ($BTC) for more reactive sentiment signals.
- **Multi-Timeframe Features**: Use features from the 1-hour and 4-hour timeframes in the 15-minute model to capture higher-level trend context.

## 2. Model Architecture
- **Probability Calibration**: Use Platt Scaling or Isotonic Regression to ensure the output probabilities (e.g., 0.7) actually correspond to a 70% win rate.
- **Temporal Convolutional Networks (TCN) or LSTMs**: Experiment with sequence-based deep learning models that can inherently capture the time-series nature of candles better than tree-based models.
- **Custom Loss Functions**: Implement a "Profit-Weighted" loss function that penalizes large misses more heavily than small ones.

## 3. Training & Validation
- **Synthetic Oversampling (SMOTE)**: Address the extreme class imbalance in short-term models by generating synthetic examples of the "UP" and "DOWN" classes.
- **Walk-Forward Bayesian Optimization**: Automate the tuning of hyperparameters (learning rate, depth) within each rolling fold.
- **Dynamic Thresholding**: Instead of a fixed 0.7% profit threshold, adapt the target based on current market volatility (ATR).

## 4. Execution & Production
- **Live Signal Dashboard**: Build a lightweight FastAPI/Streamlit dashboard to visualize the model's live predictions and confidence levels.
- **Webhook Integration**: Connect the signal generator to a Telegram or Discord bot for real-time notifications.
- **Paper Trading Engine**: Implement a simulated trading module that uses live Kraken webhooks to "trade" the signals without real capital to further validate the EV metrics.

## 5. Technical Debt & DevOps
- **Unit Testing**: Add tests for technical indicator correctness and data pipeline integrity.
- **Dockerization**: Containerize the pipeline to ensure consistent execution environments across different machines.
- **Automated CI/CD**: Set up GitHub Actions to run the evaluation pipeline whenever new features are added.
