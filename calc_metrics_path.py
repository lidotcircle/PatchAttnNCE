from i2imetrics.all_score import calculate_scores_given_paths
import click


@click.command()
@click.option('--real', help='real images path', required=True)
@click.option('--fake', help='fake images path', required=True)
@click.option('--batch_size',   help='batch size', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--device', help='torch device', default='cpu', show_default=True)
@click.option('--torch_inception',   help='pytorch inceptionv3 weights', is_flag=True)
@click.option('--inf_version',   help='FID_inf and IS_inf', is_flag=True)
def eval_metrics(
    real: str,
    fake: str,
    batch_size: int,
    device: str,
    torch_inception: bool,
    inf_version: bool
):
    results = calculate_scores_given_paths([real, fake], batch_size, device, dims=2048, use_fid_inception=not torch_inception, inf_version=inf_version)
    for vals in results:
        kid = vals['kid']
        print('[KID: %.5f (%.5f), FID: %.2f, FID_inf: %.2f, IS_inf: %.2f](%s)' % (kid[0], kid[1], vals['fid'], vals['fid_inf'], vals['is_inf'], vals['path']))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    eval_metrics() # pylint: disable=no-value-for-parameter