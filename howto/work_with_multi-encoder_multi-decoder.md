## Multi-Encoder/Multi-Decoder
In order to work with both image observations and vector observations, it is necessary to find a way to handle multiple observations. We implemented two classes: one for the encoder (`MultiEncoder`) and the other for the decoder (`MultiDecoder`). These classes are general and all the algorithms can exploit them. They take in input two parameters: the `cnn_encoder`/`cnn_decoder` and the `mlp_encoder`/`mlp_decoder`. All these 4 components must have two mandatory attributes: the `input_dim` and the `output_dim` which specify the input and output dimension of the encoder/decoder, respectively.
> [!NOTE]
>
> `cnn_encoder` and `mlp_encoder` **cannot** be both set to `None`.
>
> `cnn_decoder` and `mlp_decoder` **cannot** be both set to `None`.

Another **mandatory** attribute of the *cnn* and *mpl* encoders/decoders is the attribute `keys`. This attribute indicates the *cnn*/*mlp* keys that the encoder/decoder encodes/reconstructs.

### Multi-Encoder
The observations are encoded by the `cnn_encoder` and `mlp_encoder` and then the embeddings are concatenated on the last dimension (the *cnn* encoder encodes the observations defined by the `algo.cnn_keys.encoder` and the *mlp* encoder encodes the observations defined by the `algo.mlp_keys.encoder`). If one between the *cnn* or *mlp* encoder is not present, then the output will be equal to the output of the *mlp* or *cnn* encoder, respectively. So the `cnn_encoder` and the `mlp_encoder` must return a `Tensor`.

> [!NOTE]
>
> From our experience, we prefer to concatenate the images on the channel dimension and the vectors on the last dimension and then compute the embeddings with the `cnn_encoder` and the `mlp_encoder`, which take in input the concatenated images and the concatenated vectors, respectively.

### Multi-Decoder
The Multi-Decoder takes in input the features/states and tries to reconstruct the observations. The same features are passed in input to both the `cnn_decoder` and the `mlp_decoder`. Each of them outputs the reconstructed observations defined by the `algo.cnn_keys.decoder` and `algo.mlp_keys.decoder`, respectively. So the two decoders must return a python dictionary in the form: `Dict[key, rec_obs]`, where `key` is either a *cnn* or *mlp* key.

> [!NOTE]
>
> From our experience, we prefer to reconstruct the concatenated images with the `cnn_encoder` and then split them on the channel dimension. Instead, to reconstruct the vectors, the `mlp_encoder` is composed by two parts: *(i)* a shared backbone and *(ii)* a list of heads, one for each vector. So the features are processed by the backbone and then each head takes in input the output of the backbone to reconstruct the observation.