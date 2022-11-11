Link to FlaxBARTConditionalGeneration source code https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/bart/modeling_flax_bart.py#L1327

<pre>
import jax
import jax.numpy as jnp

class FlaxBartForConditionalGeneration(FlaxBartPreTrainedModel):
    dtype: jnp.dtype = jnp.float32
    
    ....
    def decode(
        ...
        encoder_attention_mask: <b>Optional[jnp.ndarray]</b> = None,
        decoder_attention_mask: <b>Optional[jnp.ndarray]</b> = None,
        decoder_position_ids: <b>Optional[jnp.ndarray]</b> = None,
        dropout_rng: PRNGKey = None,
    ):
    
        if encoder_attention_mask is None:
            ...
            encoder_attention_mask = <b>jnp.ones((batch_size, sequence_length))</b>
        ...
        if decoder_attention_mask is None:
            decoder_attention_mask = <b>jnp.ones((batch_size, sequence_length))</b>
        ...
        if decoder_position_ids is None:
            ...
            decoder_position_ids = <b>jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)</b>
        ...
        outputs = self.module.apply(
            inputs,
            decoder_input_ids= <b>jnp.array(decoder_input_ids, dtype="i4"</b>
            ...)
            
    def prepare_inputs_for_generation(
        ...
        attention_mask: Optional[jnp.DeviceArray] = None,
        decoder_attention_mask: Optional[jnp.DeviceArray] = None,
        ...
    ):
     
        if ...
            ...
        else:
            position_ids = <b>jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))</b>

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        ...
        
<pre>
