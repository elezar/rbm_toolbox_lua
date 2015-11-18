-- visualisation function for the rbm training process.

require('image')

function create_weight_image(W, image_dimensions, filename)
	-- Create an image from the specified weights matrix W

	w = image_dimensions[1]
	h = image_dimensions[2]
	assert(W:size(2) == w*h)

	n_filters = W:size(1)
	n_channels = 1

	pad = 1
	nrows = math.ceil(math.sqrt(n_filters))

	local weight = W:view(n_filters, n_channels, w, h)
    local filters = image.toDisplayTensor{input=weight, padding=pad,
    	nrow=nrows, scaleeach=true, symmetric=false}

    image.save(filename, filters)

end