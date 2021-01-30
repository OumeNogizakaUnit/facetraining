import click
import shutil
from pathlib import Path

from facetraining.utils import (list_categories,
                                load_image,
                                load_face_image,
                                find_all_image_path,
                                make_model,
                                save_model,
                                load_model,
                                save_categories,
                                load_categories,
                                find_face_locations,
                                predict_all_face,
                                save_result_image)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option('-o',
              '--output',
              default='model.sav',
              show_default=True,
              help='学習済みモデル出力先')
@click.argument('image_base_dir',
                type=click.Path(exists=True))
def learn(output, image_base_dir):
    '''
    IMGAE_BSAE_DIR に指定した画像を用いて学習済みモデルの作成
    '''
    pathlist = find_all_image_path(image_base_dir)
    categories = list_categories(image_base_dir)
    click.echo(f'{categories}')
    svc, score = make_model(pathlist, categories)
    click.echo(f'learn score: {score}%')
    save_model(svc, output)
    click.echo(f'save module: {output}')
    save_categories(categories, 'member.csv')
    click.echo('finish!!!')


@cli.command()
@click.option('-m',
              '--model',
              default='model.sav',
              show_default=True,
              type=click.Path(exists=True),
              help='学習済みモデルファイル')
@click.option('-l',
              '--memberlist',
              default='member.csv',
              show_default=True,
              type=click.Path(exists=True),
              help='メンバー名のCSV')
@click.option('-o',
              '--output',
              help='結果画像の出力',
              type=click.Path())
@click.argument('pred_image_path',
                type=click.Path(exists=True))
def predict(model, memberlist, pred_image_path, output):
    '''
    学習済みモデルを使って予測
    '''
    svc = load_model(model)
    image = load_image(pred_image_path)
    categories = load_categories(memberlist)
    face_locations = find_face_locations(image)
    result_list = predict_all_face(svc, face_locations, image)
    for result in result_list:
        rate, member = max(zip(result, categories))
        click.echo(f'{member}: {round(rate*100, 2)}')
    if output is not None:
        save_result_image(image,
                          output,
                          face_locations,
                          result_list,
                          categories)
        click.echo(f'output: {output}')


@cli.command()
@click.option('-m',
              '--model',
              default='model.sav',
              show_default=True,
              type=click.Path(exists=True),
              help='学習済みモデルファイル')
@click.option('-l',
              '--memberlist',
              help='メンバー名のCSV')
@click.option('-o',
              '--output-base-dir',
              type=click.Path(),
              required=True,
              help='出力先のディレクトリ')
@click.option('--move',
              help='copyではなくmoveする',
              is_flag=True,
              default=False,
              show_default=True)
@click.argument('image_base_dir',
                type=click.Path(exists=True))
def sort(model, memberlist, output_base_dir, image_base_dir, move):
    '''
    学習済みモデルをつかって画像を仕分ける
    '''
    svc = load_model(model)
    pathlist = find_all_image_path(image_base_dir)
    categories = load_categories(memberlist)
    for path in pathlist:
        click.echo(f'{path}')
        cnv_list = load_face_image(path)
        result = svc.predict_proba(cnv_list)[0]
        rate, member = max(zip(result, categories))
        click.echo(f'{round(rate*100, 2)}%: {member}')
        todir = Path(output_base_dir, member)
        if todir.exists() is False:
            click.echo(f'mkdir {todir}')
            todir.mkdir(parents=True)
        fromdirname = path.parent.name
        tofilename = f'{fromdirname}_{member}_{path.name}'
        topath = Path(todir, tofilename)
        click.echo(f'{path} -> {topath}')
        shutil.copy(path, topath)
