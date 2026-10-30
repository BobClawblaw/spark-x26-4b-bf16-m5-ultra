| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| soak | 60/60 | 100.0% | 114s | 3 chats x 20 turns at once beside a decoding stream (5.8k-6.5k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| drafts | 12/12 | 100.0% | 33s | drafted 99 tok/s vs undrafted 99 tok/s |
| concurrent | 12/12 | 100.0% | 23s | 16 streams together vs alone |
| streaming | 4/4 | 100.0% | 11s | identical |
| multiturn | 7/8 | 87.5% | 7s | recalled 7/8 facts |
| stop | 3/3 | 100.0% | 1s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 18s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 1s | 3/3 exact |
