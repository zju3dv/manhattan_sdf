from lib.config import args, cfg


def run_mesh_extract():
    from lib.datasets import make_data_loader
    from lib.networks import make_network
    from lib.utils.mesh_utils import extract_mesh, refuse, transform
    from lib.utils.net_utils import load_network
    import open3d as o3d

    network = make_network(cfg).cuda()
    load_network(
        network,
        cfg.trained_model_dir,
        resume=cfg.resume,
        epoch=cfg.test.epoch
    )
    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)

    mesh = extract_mesh(network.model.sdf_net)
    mesh = refuse(mesh, data_loader)
    mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)

    assert args.output_mesh != ''
    o3d.io.write_triangle_mesh(args.output_mesh, mesh)


def print_result(result_dict):
    for k, v in result_dict.items():
        print(f'{k:7s}: {v:1.3f}')


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    from lib.networks import make_network
    from lib.utils.mesh_utils import extract_mesh, refuse, transform
    from lib.utils.net_utils import load_network
    import open3d as o3d

    network = make_network(cfg).cuda()
    load_network(
        network,
        cfg.trained_model_dir,
        resume=cfg.resume,
        epoch=cfg.test.epoch
    )
    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)

    mesh = extract_mesh(network.model.sdf_net)
    mesh = refuse(mesh, data_loader)
    mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)

    if args.output_mesh != '':
        o3d.io.write_triangle_mesh(args.output_mesh, mesh)

    mesh_gt = o3d.io.read_triangle_mesh(f'{cfg.test_dataset.data_root}/{cfg.test_dataset.scene}/gt.obj')
    evaluate_result = evaluator.evaluate(mesh, mesh_gt)
    print_result(evaluate_result)


if __name__ == '__main__':
    globals()['run_' + args.type]()
