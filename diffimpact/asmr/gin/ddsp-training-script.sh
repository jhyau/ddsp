clips=" "air-puffer-2" "tin" "vinyl-surface" "pink-pencil-holder" "plastic-surface" "leather-bag-2" "silicon" "foam" "glitter-tubes" "wood-surface" "rhinestone-surface-2" "coconut-shell" "plastic-surface-2" "stone-floor-tile" "rode-nt1-mic" "wood-surface-2" "steel" "cork" "silicon-surface-2" "silicon-surface" "rubber" "diamond" "diamond-2" "ceramic" "studded-ball-2" "acrylic" "plastic" "studded-ball" "3dio-ear" "air-puffer" "pumice-stone" "glass-jar-2" "soap-2" "lollipop-2" "vinyl-surface-2" "zoom-h4n-mic-2" "soap" "wood" "vinyl" "basketball-2" "cardboard" "pink-pencil-holder-2" "basketball" "leather" "phone-2" "phone" "zoom-h4n-mic" "rode-nt1-mic-2" "3dio-ear-2" "plastic-mold" "glass" "glitter-tubes-2" "rhinestone-surface" "velvet-fabric" "leather-bag" "plastic-mold-2" "glass-jar" "lollipop" "stone_floor-tile-2" "marble" "

for entry in $clips
	do
		path="/juno/u/jyau/regnet/ddsp/diffimpact/asmr/regnet-labels/3hr-"
		path+=$entry
		if [ ! -d "$path" ]; then
			echo $entry
			echo $path
			python /juno/u/jyau/regnet/ddsp/diffimpact/asmr/train.py \
			--model-type diffimpact \
			--train-pattern "/juno/u/jyau/regnet/data/features/ASMR/orig_asmr_by_material_clips/audio_10s_44100hz_ddsp/3hr/${entry}/train/*.wav" \
		        --validation-pattern "/juno/u/jyau/regnet/data/features/ASMR/orig_asmr_by_material_clips/audio_10s_44100hz_ddsp/3hr/${entry}/val/*.wav" \
			--save-dir /juno/u/jyau/regnet/ddsp/diffimpact/asmr/regnet-labels/3hr-${entry}

			wait $! # Used to wait for background process, if want to send to background (need to have & added to end of command)
		fi
	done
	
