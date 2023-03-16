ffmpeg -r 1 -start_number 1 -i Data/Outputs/Crack_Masks/out%04d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
