#!/usr/bin/env bash
set -euo pipefail

cd /home/xavier/scout-edge
mkdir -p media
curl -fL --retry 2 'https://npgallery.nps.gov/GetAsset/d2f72765-6363-4fe6-bcfa-483f41091a38/proxymdres?' \
  -o media/acadia-park-staff.mp4
curl -fL --retry 2 'https://www.nps.gov/nps-audiovideo/audiovideo/0771438d-5fc2-4839-8ddc-723a3ba77137480p.mp4' \
  -o media/grand-canyon-guided-hike-1.mp4
curl -fL --retry 2 'https://www.nps.gov/nps-audiovideo/audiovideo/5d878af8-d70f-4fd3-aaff-e49d746a9874480p.mp4' \
  -o media/grand-canyon-guided-hike-2.mp4
