# Table: Manual tracing parameters
| Name | Value | Symbol | Description | Source | 
|  ---- | ---- | ---- | ---- | ---- |
| `manual_trace_on` | 0 | - | Turn on manual tracing (0=no, 1=yes) | - |
| `manual_trace_time_on` | 10000 | - | Time (days) after which manual tracing is turned on | - |
| `manual_trace_on_hospitalization` | 1 | - | Trace when hospitalized if tested positive (no effect if manual_trace_on_positive is on) | - |
| `manual_trace_on_positive` | 0 | - | Trace when hospitalized if tested positive (no effect if manual_trace_on_positive is on) | - |
| `manual_trace_delay` | 1 | - | Delay (days) between triggering manual tracing due to testing/hospitalization and tracing occurring | - |
| `manual_trace_exclude_app_users` | 0 | - | Whether or not to exclude app users when performing manual tracing (exclude=1, include=0) | - |
| `manual_trace_n_workers` | 300 | - | Number of Contact Tracing Workers | NACCHO Position Statement, 2020 |
| `manual_trace_interviews_per_worker_day` | 6 | - | Number of interviews performed per worker per day | https://www.gwhwi.org/estimator-613404.html |
| `manual_trace_notifications_per_worker_day` | 12 | - | Number of trace notifications performed per worker per day | https://www.gwhwi.org/estimator-613404.html |
| `manual_traceable_fraction_household` | 1 | - | The fraction of household contacts that can be successfully traced | - |
| `manual_traceable_fraction_occupation` | 0.8 | - | The fraction of occupation contacts that can be successfully traced | - |
| `manual_traceable_fraction_random` | 0.05 | - | The fraction of random contacts that can be successfully traced | - |