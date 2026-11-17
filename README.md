# Will There Be a Divvy Bike? (Loyola)

A class project by Jonathan Afari Takyi, Nick, and Daniel (Loyola University Chicago).

Divvy riders near Loyola's Lake Shore Campus can see how many bikes a station has **now**, but not whether it will still have one when they get there. This project collects Divvy's live station data and trains a model to predict the chance a station runs empty in the next 30 minutes.

## What's running

| Part | Status | Folder |
|---|---|---|
| Live data collector (every 15 min) | Running | `collector/`, `.github/workflows/collect.yml` |
| Demand + empty-risk models | In progress | `model/` |
| Web app | In progress | `app/` |

## Data

- **Live:** Divvy GBFS feeds ([station status](https://gbfs.lyft.com/gbfs/1.1/chi/en/station_status.json), free bikes, station information), saved every 15 minutes to `data/`.
  - `data/station_status/YYYY-MM-DD.csv.gz`: one row per station per snapshot (bikes, e-bikes, open docks).
  - `data/free_bikes/YYYY-MM-DD.csv.gz`: dockless e-bikes near Loyola per snapshot.
  - `data/station_information.csv`: station names, locations, capacity.
- **History:** [Divvy trip data](https://divvy-tripdata.s3.amazonaws.com/index.html), used under the [Divvy Data License Agreement](https://www.divvybikes.com/data-license-agreement).

Snapshot times are in UTC.

## License

MIT
