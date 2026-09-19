#!/bin/bash
# Transfert sécurisé vers Windows — SANS clés privées
rsync -av --exclude-from=/home/kali/deepfake_detector/.rsync-exclude \
    /home/kali/deepfake_detector/ \
    /media/sf_Kali-share/DeepfakeDetector/source/
echo "Transfert terminé (clés et DB exclus)"
