# Evaluation Analysis

**Source:** ../eval_results.json

## Results

|Setup|Success %|Tokens/turn|Latency/turn (ms)|
|---|---|---|---|
|no_memory|7.5|47|0.00|
|naive|85.0|52|0.00|
|agent_mem_db|92.5|56|0.13|

## Latency Percentiles (per-task ms)

|Setup|p50|p95|p99|
|---|---|---|---|
|no_memory|0.00|0.00|0.00|
|naive|0.00|0.00|0.01|
|agent_mem_db|0.24|8.16|8.80|

## By Task Type

|Setup|Short success %|Long success %|
|---|---|---|
|no_memory|9.7|0.0|
|naive|100.0|33.3|
|agent_mem_db|100.0|66.7|
