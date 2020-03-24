import cv2
import click
import os


@click.command()
@click.option('-i', '--indir', required=True, help='Input figure directory')
@click.option('-o', '--output', type=str, default=None, help='Output video path')
@click.option('-r', '--rate', type=int, default=10, help='Playback rate (images per second)')
def main(indir, output, rate):
    images = [img for img in os.listdir(indir) if img.endswith('.jpg') or img.endswith('.png')]
    images = list(sorted(images, key=lambda v: int(v.split('.')[0])))
    frame = cv2.imread(os.path.join(indir, images[0]))
    height, width, layers = frame.shape

    if output is None:
        output = os.path.dirname(indir) + '.mp4'
    output = os.path.splitext(output)[0] + '.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output, fourcc, rate, (width, height))

    for image in images:
        frame_num = image.split('.')[0]
        watermark = f'Frame: {frame_num}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (70, 70)
        fontScale = 1.5
        color = (255, 0, 0)
        thickness = 2

        image = cv2.imread(os.path.join(indir, image))
        image = cv2.putText(image, watermark, org, font, fontScale, color, thickness, cv2.LINE_AA)
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main()
