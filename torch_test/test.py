import torch


def compute_normals_loss(superpixel, normals):
    """
    compute the loss with superpixel information.
    Forces normal vectors in one superpixel to be equal.

    :param superpixel: superpixel labels
    :param normals: vector normals
    :return: normals loss
    """

    # init normals loss
    normals_loss = 0

    # shape of normals should be batchsize, 3, h, w
    # shape of superpixel should be batchsize, 1, h, w

    # change shape from batchsize, 1, h, w to batchsize, h * w
    superpixel = superpixel.view(superpixel.shape[0], superpixel.shape[2] * superpixel.shape[3])

    # shape from (batch, 3, h, w) to (batch, 3, h * w)
    normals = normals.view(normals.shape[0], normals.shape[1], normals.shape[2] * normals.shape[3])

    # get indices
    min_idx = torch.min(superpixel).item()
    max_idx = torch.max(superpixel).item() + 1

    superpixel_indices = torch.arange(start=min_idx, end=max_idx).cuda()

    # get pixel for each superpixel area
    #zeros = torch.zeros_like(superpixel, dtype=torch.bool).cuda()
    #ones = torch.ones_like(superpixel, dtype=torch.b).cuda()

    all_coords = []

    for i in superpixel_indices:
        # set all pixel which are not at current index to zero
        coords = superpixel == i

        # get all nonzero coordinates
        #coords1 = torch.nonzero(coords)

        #rint(coords1.shape)
        # shape is (num of nonzero, 3)
        # convert to (num of nonzero, 1, 3) length to be able to stack on dim 1
        #coords = torch.unsqueeze(coords, dim=1)

        # add current tensor to list
        all_coords.append(coords)

        normals_per_superpixel = torch.masked_select(normals[:, :], )

    # stack all tensors in the list on dim=1
    # --> shape will be (batchsize, superpixel_index, h*w)
    # Note: still take all values here to have equal shapes
    # Do non zero later
    coords_tensor = torch.stack(all_coords, dim=1).cuda()

    # TODO try torch index select

    test = torch.nonzero(coords_tensor)

    for j in range(coords_tensor.shape[1]):
        indices = coords_tensor[:, j]

        # get all normals pixel values per superpixel area
        normals_per_superpixel = torch.masked_select(normals[:, :], indices)


    # normals_per_superpixel = torch.tensor([normals[:, idx] for idx in coords])

    std = torch.std(normals_per_superpixel)
    for normals_in_one_superpixel_area in normals_per_superpixel:
        # calculate standard deviation for each area
        # calculate first for each channel of current area, then sum for current area

        std = torch.std()
        normals_loss += np.sum(np.std(normals_in_one_superpixel_area, axis=1))

    return normals_loss


if __name__ == "__main__":

    superpixel = torch.randint(0, 711, size=(1, 1, 19, 64)).cuda()
    normals = torch.randint(0, 111, size=(1, 3, 19, 64)).cuda()

    loss = compute_normals_loss(superpixel, normals)
