#!/bin/bash

pip install -r requirements.txt
if [ $# -lt 2 ]; then
    echo "Usage: $0 <game> <root_folder>"
    exit 1
fi
game=$1
root_folder=$2

if [ ! -d "$root_folder" ]; then
    mkdir -p "$root_folder"
    echo "Directory created: $root_folder"
else
    echo "Directory already exists: $root_folder"
fi

if [ "$game" == "cave" ]; then
    python3 ./sturgeon/input2tile.py --outfile $root_folder/cave.tile --textfile ./sturgeon/levels/kenney/cave.lvl --imagefile ./sturgeon/levels/kenney/cave.png
    python3 ./sturgeon/tile2scheme.py --outfile $root_folder/cave.scheme --tilefile $root_folder/cave.tile --count-divs 1 1 --pattern nbr-plus
fi
if [ "$game" == "platform" ]; then
    python3 ./sturgeon/input2tile.py --outfile $root_folder/platform.tile --textfile ./sturgeon/levels/kenney/platform.lvl --imagefile ./sturgeon/levels/kenney/platform.png
    python3 ./sturgeon/tile2scheme.py --outfile $root_folder/platform.scheme --tilefile $root_folder/platform.tile --count-divs 1 1 --pattern ring
fi
if [ "$game" == "slide" ]; then
    python3 ./sturgeon/input2tile.py --outfile $root_folder/slide.tile --textfile ./levels/slide.lvl --imagefile ./levels/slide.png
    python3 ./sturgeon/tile2scheme.py --outfile $root_folder/slide.scheme --tilefile $root_folder/slide.tile --pattern nbr-plus
fi
if [ "$game" == "vertical" ]; then
    python3 ./sturgeon/input2tile.py --outfile $root_folder/vertical.tile --textfile ./sturgeon/levels/kenney/vertical.lvl --imagefile ./sturgeon/levels/kenney/vertical.png
    python3 ./sturgeon/tile2scheme.py --outfile $root_folder/vertical.scheme --tilefile $root_folder/vertical.tile --count-divs 3 1 --pattern nbr-plus
fi
if [ "$game" == "sokoban" ]; then
    python3 ./sturgeon/input2tile.py --outfile $root_folder/sokoban.tile --textfile ./sturgeon/levels/mkiii/soko.lvl
    python3 ./sturgeon/tile2scheme.py --outfile $root_folder/sokoban.scheme --tilefile $root_folder/sokoban.tile
fi
if [ "$game" == "cavedoors" ]; then
    python3 ./sturgeon/input2tile.py --outfile $root_folder/cave-doors.tile --textfile ./sturgeon/levels/kenney/cave-doors.lvl --imagefile ./sturgeon/levels/kenney/cave-doors.png
    python3 ./sturgeon/tile2scheme.py --outfile $root_folder/cave-doors.scheme --tilefile $root_folder/cave-doors.tile --count-divs 1 1 --pattern nbr-plus
fi
if [ "$game" == "caveportal" ]; then
    python3 ./sturgeon/input2tile.py --outfile $root_folder/cave-junction.tile --textfile ./sturgeon/levels/kenney/cave-junction.lvl --imagefile ./sturgeon/levels/kenney/cave-junction.png
    python3 ./sturgeon/tile2scheme.py --outfile $root_folder/cave-junction.scheme --tilefile $root_folder/cave-junction.tile --count-divs 1 1 --pattern nbr-plus
fi

RANDOM=$$
for ((i=1; i<=10000; i++)); do
    if [ "$game" == "cave" ]; then
        python3 ./sturgeon/scheme2output.py --random $RANDOM --outfile $root_folder/$i --schemefile $root_folder/cave.scheme \
        --size 16 32 --pattern-hard --count-soft --reach-start-goal br-tl 5 --reach-move maze \
        --solver pysat-rc2 pysat-rc2-boolonly clingo-be scipy --out-tlvl-none --out-result-none

        python3 ./sturgeon/level2repath.py --outfile $root_folder/$i-path.lvl --textfile $root_folder/$i.lvl --reach-move maze
    fi
    if [ "$game" == "platform" ]; then
        python3 ./sturgeon/scheme2output.py --random $RANDOM --outfile $root_folder/$i --schemefile $root_folder/platform.scheme \
        --size 16 32 --pattern-hard --count-soft --reach-start-goal l-r 6 --reach-move platform \
        --solver pysat-rc2 pysat-rc2-boolonly clingo-be scipy --out-tlvl-none --out-result-none

        python3 ./sturgeon/level2repath.py --outfile $root_folder/$i-path.lvl --textfile $root_folder/$i.lvl --reach-move platform
    fi
    if [ "$game" == "slide" ]; then
        python3 ./sturgeon/scheme2output.py --random $RANDOM --outfile $root_folder/$i --schemefile $root_folder/slide.scheme \
        --size 32 16 --pattern-hard  --reach-start-goal b-t 6 --reach-move tomb \
        --solver pysat-rc2 pysat-rc2-boolonly clingo-be scipy --out-tlvl-none --out-result-none

        python3 ./sturgeon/level2repath.py --outfile $root_folder/$i-path.lvl --textfile $root_folder/$i.lvl --reach-move tomb
    fi
    if [ "$game" == "vertical" ]; then
        python3 ./sturgeon/scheme2output.py --random $RANDOM --outfile $root_folder/$i --schemefile $root_folder/vertical.scheme \
        --size 20 16 --pattern-hard --count-soft --reach-start-goal b-t 6 --reach-move supercat-new \
        --solver pysat-rc2 pysat-rc2-boolonly clingo-be scipy --out-tlvl-none --out-result-none

        python3 ./sturgeon/level2repath.py --outfile $root_folder/$i-path.lvl --textfile $root_folder/$i.lvl --reach-move supercat-new
    fi
    if [ "$game" == "sokoban" ]; then
        python3 ./sturgeon/scheme2output.py --random $RANDOM --outfile $root_folder/$i --schemefile $root_folder/sokoban.scheme \
        --mkiii-example soko --size 12 12 --mkiii-layers 15 \
        --solver pysat-minicard --out-tlvl-none --out-result-none
        python3 ./level2image/level2image.py $root_folder/$i.lvl --fmt png --tile-image-folder sprites/sokoban
    fi
    if [ "$game" == "cavedoors" ]; then
        python3 ./sturgeon/scheme2output.py  --random $RANDOM --outfile $root_folder/$i --schemefile $root_folder/cave-doors.scheme \
        --size 16 16 --pattern-hard --count-soft \
        --reach-junction { tl 4 \
        --reach-junction } br 4 \
        --reach-junction B bl 8 \
        --reach-junction b tr 8 \
        --custom text-exclude Y hard \
        --custom text-exclude y hard \
        --custom text-exclude 1 hard \
        --custom text-exclude 2 hard \
        --reach-connect "--src { --dst b --move maze" \
        --reach-connect "--src b --dst B --move maze" \
        --reach-connect "--src B --dst } --move maze" \
        --reach-connect "--src { --dst } --move maze --unreachable" \
        --reach-connect "--src b --dst } --move maze --unreachable" \
        --solver pysat-rc2 pysat-rc2-boolonly clingo-be scipy --out-tlvl-none --out-result-none

        python3 ./sturgeon/level2repath.py --outfile $root_folder/$i-path.lvl --textfile $root_folder/$i.lvl \
            --reach-connect "--src { --dst b --move maze" \
            --reach-connect "--src b --dst B --move maze" \
            --reach-connect "--src B --dst } --move maze" \
            --reach-connect "--src { --dst } --move maze --unreachable" \
            --reach-connect "--src b --dst } --move maze --unreachable" 
    fi
    if [ "$game" == "caveportal" ]; then
        python3 ./sturgeon/scheme2output.py --random $RANDOM --outfile $root_folder/$i --schemefile $root_folder/cave-junction.scheme \
        --size 16 16 --pattern-hard --count-soft \
        --reach-junction 0 tl 4 --reach-junction 1 tr 4 --reach-junction 2 bl 4 --reach-junction 3 br 4 \
        --reach-connect "--src 0 --dst 1 --move maze" \
        --reach-connect "--src 2 --dst 3 --move maze" \
        --reach-connect "--src 0 --dst 2 --move maze --unreachable" \
        --solver pysat-rc2 pysat-rc2-boolonly clingo-be scipy --out-tlvl-none --out-result-none

        python3 ./sturgeon/level2repath.py --outfile $root_folder/$i-path.lvl --textfile $root_folder/$i.lvl \
        --reach-connect "--src 0 --dst 1 --move maze" \
        --reach-connect "--src 2 --dst 3 --move maze" \
        --reach-connect "--src 0 --dst 2 --move maze --unreachable"
    fi
done