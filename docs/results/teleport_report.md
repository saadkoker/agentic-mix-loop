# Mix report: teleport

Stopped: plateau: 8 rounds without an accepted change  
Rounds: 14, accepted changes: 3, LLM spend: $3.23, wall-clock 53.0 min  
Phase 5, agent llm, acceptance margin 0.0167 (judge noise spread 0.0142)

## Scores

| | score | PQ | CE | worst window | PC | CU |
|---|---|---|---|---|---|---|
| flat baseline | 9.3346 | 8.185 | 7.162 | 7.283 @ 5s | 6.452 | 7.778 |
| best mix | 9.4596 | 8.208 | 7.278 | 7.652 @ 175s | 6.555 | 7.864 |

Change vs baseline: **+0.1250**

## Files

- `renders/final/1_flat_baseline.wav` (flat sum through the fixed limiter)
- `renders/final/2_best_mix.wav` (best mix at -14.0 LUFS)
- `renders/final/3_best_mix_loud.wav` (final limiter pushed to -9.7 LUFS, -1.05 dBTP)
- `chains/final.json` (rebuild exactly with `python mixloop.py score teleport --chain chains/final.json`)

![score history](../img/teleport_score_history.png)

## Accepted changes

- round 1 `*balance*` +0.0339 -> 9.3684: Follow the measured sensitivities: beat down 3 dB, lead vox 1 up 3 dB, small lifts on LeadVox2, BackingVox1 and BackingVox4. Lead and secondary vocals stay centred, backing vocals get mild opposing pans for space. The beat keeps a slightly narrowed stereo image.
- round 2 `02_LeadVox1` +0.0673 -> 9.4357: Preferred B: strongest audition (+0.0673); duck intro outtakes, retain hooks 2 dB lower, and add gentle cleanup/dynamics. Worst-window quality improved to 7.591; no added ambience.
- round 6 `bus:VoxVerb` +0.0238 -> 9.4596: Best audition: section-ridden wet-only plate-style vocal space, 35 ms pre-delay, 250 Hz–8 kHz return filtering. Intro/chorus lift and quieter rap verses improved CE +0.0809 and overall +0.0238, passing first bar; retest pending. Delay omitted due to one-space-plugin limit.

## Final chains

- **01_Beat**: gain_pan(gain_db=-3, pan=0, width=0.95)
- **02_LeadVox1**: eq(hp_hz=85, low_shelf_hz=120, low_shelf_db=0, peak1_hz=300, peak1_db=0, peak1_q=1, peak2_hz=3000, peak2_db=0, peak2_q=1, high_shelf_hz=8000, high_shelf_db=0, lp_hz=20000) -> deess(freq_hz=6000, threshold_db=-36, ratio=2.5, range_db=3, attack_ms=1.5, release_ms=65) -> comp(threshold_db=-25, ratio=2, attack_ms=20, release_ms=110, makeup_db=1.2, mix=1) -> gain_pan(gain_db=auto[linear 0s:1>11.15s:1>11.35s:-35>15.45s:-35>15.7s:3], pan=0, width=1)
- **03_LeadVox2**: gain_pan(gain_db=1.5, pan=0, width=1)
- **04_BackingVox1**: gain_pan(gain_db=1, pan=-20, width=1)
- **05_BackingVox2**: gain_pan(gain_db=0, pan=20, width=1)
- **06_BackingVox3**: gain_pan(gain_db=-1, pan=-30, width=1)
- **07_BackingVox4**: gain_pan(gain_db=1.5, pan=30, width=1)
- **bus VoxVerb: inputs [-] sends [02_LeadVox1@-16dB, 03_LeadVox2@-16dB, 04_BackingVox1@-14dB, 05_BackingVox2@-14dB, 06_BackingVox3@-14dB, 07_BackingVox4@-14dB] chain reverb(room_size=0.55, damping=0.6, width=0.9, predelay_ms=35, lowcut_hz=250, highcut_hz=8000, mix=1) -> eq(hp_hz=250, low_shelf_hz=120, low_shelf_db=0, peak1_hz=300, peak1_db=0, peak1_q=1, peak2_hz=3000, peak2_db=0, peak2_q=1, high_shelf_hz=8000, high_shelf_db=0, lp_hz=8000) -> gain_pan(gain_db=auto[linear 0s:2>15.5s:2>17s:-1>45s:-1>47s:-4>76s:-4>78s:2>105.5s:2>...], pan=0, width=1)**
- **master**: (empty) -> fixed limiter -1.0 dBTP
