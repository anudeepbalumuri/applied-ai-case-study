# Capture protocol (Route 2: stock apps) — one page

You need an iPhone 15 or newer, a charged battery, and the lights on. Pick **one** tier.
Do not move furniture or close or open doors while you capture.

## Before you start, all tiers
- Turn on every light and open the blinds. Very dark rooms produce no result.
- Walk slowly, about half your normal walking speed. Turn slowly too: one full turn should take about 10 seconds.
- Keep fingers off the lens.
- Hold the phone the same way the whole time: upright (portrait) or sideways (landscape) are both fine.

## Tier A: LiDAR (iPhone 15 Pro / Pro Max, 16 Pro / Pro Max or newer)
1. **Install:** App Store → **"Stray Scanner"** (free, publisher Stray Robots). Open it and allow Camera access.
2. **Start:** stand just inside the front door of the area. Press the red record button.
3. **Walk:**
   - Go into every room.
   - In each room, walk along the walls at about arm's length from them, pointing the phone at the wall at chest height.
   - Then tilt up for 2 seconds so the ceiling is in view, and tilt down for 2 seconds to see the floor.
   - Walk through each doorway slowly, pointing the phone through it.
   - End where you started.
4. **How long:** about 1 minute per room. 10 minutes is the maximum.
5. **Avoid:** pointing at bright windows for more than 2 seconds, running, and covering the back camera.
6. **Stop and export:** press stop. In the app's list, tap the recording → Share → **Save to Files**. Then copy that folder to the computer (cable, AirDrop or iCloud Drive). Do not rename the files inside it.

## Tier B: Video (any iPhone 15 or newer)
1. **Settings:** Settings → Camera → Record Video → **1080p at 30 fps**, **HDR Video OFF**. Settings → Camera → Formats → **Most Compatible**.
2. **Camera app, Video mode, 1x zoom.** Hold the phone at chest height.
3. **Walk:** the same path as Tier A. In each room, stand in the middle and turn slowly through a full circle. Then walk once around the room facing the walls. Pass slowly through every doorway.
4. **How long:** 30–60 seconds per room, 8 minutes maximum. One continuous clip; do not pause.
5. **Avoid:** fast pans (they blur and break tracking), zooming, and filming a mirror head-on.
6. **Hand-off:** AirDrop the `.mov` to the computer.

## Tier C: Photos (any iPhone 15 or newer), 2–8 photos per room
1. **Camera app, Photo mode, 1x, landscape (sideways).** Turn off Live Photos and Portrait mode.
2. **For each room, in this order:**
   1. **Entry photo:** stand in the doorway you came in by and photograph the whole room.
   2. **Sweep:** stand in a corner. Take photos while turning, so each photo shows about one third of the previous one. Usually 4–5 photos.
   3. **Doorway photo:** before leaving, photograph the doorway you will leave by from about 2 m away, so the next room is visible through it.
   - Every wall and every doorway of the room must appear in at least one photo.
   - Keep the phone level, at chest height. Don't tilt it to the ceiling.
3. **Folders:** AirDrop or copy the photos to the computer, one folder per room, named after the room: `kitchen/`, `hall/`, `bed1/` …
   - A small hallway or connector counts as a room and gets its own folder.
   - HEIC and JPEG are both fine; don't edit or crop.

## Run (all tiers): one command
```
python -m roomscan <folder-or-video>        # tier is detected automatically
```
Results go to `out/<name>/`: `plan.json` (all measurements with 95 % intervals), plus `plan.png` and `plan.svg` (the floor plan).

## Troubleshooting

| Symptom | What to do |
|---|---|
| "no layout recovered" for a photo room | Retake that room with more overlap between photos. |
| Room shown as *partial* | Some walls were never in view. Re-shoot them. |
| Very wide intervals | Thin data for that measurement. Use the LiDAR tier if precision matters. |
