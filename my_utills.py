# Imports from external libraries
import cv2
import numpy as np
# Imports from internal libraries
import configs as cfg

def convergence_map(res,name_postfix):
    color_map = {}
    groups_map = {}
    group_color_maps = {}

    colors = cfg.colors
    colors = [cfg.hex_to_rgb(x) for x in colors]

    limit = 4
    j = 10

    for k in res:
        if len(res[k]) > limit:
            if len(groups_map) == 0:
                groups_map[k] = [k]
            else:
                found_similar = False
                for m in groups_map:
                    d = np.abs(m[0] - k[0]) + np.abs(m[1] - k[1])
                    if d < 5:
                        groups_map[m].append(k)
                        found_similar = True

                if not found_similar:
                    groups_map[k] = [k]

    for g in groups_map:
        group_color_maps[g] = colors[j]
        j += 1

    for g in groups_map:
        similar = groups_map[g]
        for s in similar:
            color_map[s] = group_color_maps[g]

    vis = np.zeros(100 * 100 * 3).reshape((100, 100, 3))
    for k in res:
        if len(res[k]) > limit:
            print(f"{k},lem: {len(res[k])}")
            vis[k[0]][k[1]] = color_map[k]
            for start in res[k]:
                vis[start[0]][start[1]] = color_map[k]

    # cv2.imshow("test_vis", vis)
    cv2.imwrite(f"{cfg.result_path}colormap{name_postfix}.jpg",vis)
    # cv2.waitKey(0)
    # cv2.destroyWindow('test_vis')



