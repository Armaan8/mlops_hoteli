from datetime import datetime

def log_results(occ_class, rew_class, final_price, occ_metrics, rew_metrics, price_metrics):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
🕒 Timestamp: {timestamp}
----------------------------------------
✅ Occupancy Class: {occ_class}
✅ Points Class: {rew_class}
✅ Final Room Price: ${final_price:.2f}

📊 Occupancy Metrics: {occ_metrics}
📊 Rewards Accuracy: {rew_metrics['accuracy']}
📊 Pricing Metrics: {price_metrics}

========================================
"""
    with open("logs/pipeline_log.txt", "a") as f:
        f.write(log_entry)
