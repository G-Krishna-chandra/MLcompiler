// One TinyLlama-1.1B decoder layer in the mlc dialect.
// Shapes match TinyLlama-1.1B: hidden=2048, heads=32, kv_heads=4,
// head_dim=64, ffn=5632, vocab=32000.
//
// Verifies the dialect can express the canonical pre-norm transformer block:
//   y1 = norm(x);           q,k,v = matmul(y1, w_{q,k,v})
//   a  = attention(q,k,v);  o = matmul(a, w_o);  m = add(x, o)
//   y2 = norm(m);           f = feedforward(y2, gate, up, down);  out = add(m, f)

module {
  func.func @tinyllama_layer(
      %x:        tensor<1x2048xf16>,
      %g1:       tensor<2048xf16>,
      %g2:       tensor<2048xf16>,
      %positions: tensor<1xi32>,
      %w_q:      tensor<2048x2048xi8>,
      %w_k:      tensor<2048x256xi8>,
      %w_v:      tensor<2048x256xi8>,
      %w_o:      tensor<2048x2048xi8>,
      %w_gate:   tensor<2048x5632xi8>,
      %w_up:     tensor<2048x5632xi8>,
      %w_down:   tensor<5632x2048xi8>
  ) -> tensor<1x2048xf16> {
    %n1 = mlc.norm %x, %g1 {target_device = #mlc.device<auto>, epsilon = 1.000000e-05 : f32}
        : (tensor<1x2048xf16>, tensor<2048xf16>) -> tensor<1x2048xf16>

    %q = mlc.matmul %n1, %w_q
        {target_device = #mlc.device<auto>, quant_format = "q4_0"}
        : (tensor<1x2048xf16>, tensor<2048x2048xi8>) -> tensor<1x2048xf16>
    %k = mlc.matmul %n1, %w_k
        {target_device = #mlc.device<auto>, quant_format = "q4_0"}
        : (tensor<1x2048xf16>, tensor<2048x256xi8>) -> tensor<1x256xf16>
    %v = mlc.matmul %n1, %w_v
        {target_device = #mlc.device<auto>, quant_format = "q4_0"}
        : (tensor<1x2048xf16>, tensor<2048x256xi8>) -> tensor<1x256xf16>

    %a = mlc.attention %q, %k, %v, %positions
        {target_device = #mlc.device<auto>,
         num_heads = 32 : i64, num_kv_heads = 4 : i64, head_dim = 64 : i64}
        : (tensor<1x2048xf16>, tensor<1x256xf16>, tensor<1x256xf16>, tensor<1xi32>)
          -> tensor<1x2048xf16>

    %o = mlc.matmul %a, %w_o
        {target_device = #mlc.device<auto>, quant_format = "q4_0"}
        : (tensor<1x2048xf16>, tensor<2048x2048xi8>) -> tensor<1x2048xf16>
    %m = mlc.add %x, %o {target_device = #mlc.device<auto>} : tensor<1x2048xf16>

    %n2 = mlc.norm %m, %g2 {target_device = #mlc.device<auto>, epsilon = 1.000000e-05 : f32}
        : (tensor<1x2048xf16>, tensor<2048xf16>) -> tensor<1x2048xf16>

    %f = mlc.feedforward %n2, %w_gate, %w_up, %w_down
        {target_device = #mlc.device<auto>, quant_format = "q4_0"}
        : (tensor<1x2048xf16>, tensor<2048x5632xi8>, tensor<2048x5632xi8>, tensor<5632x2048xi8>)
          -> tensor<1x2048xf16>

    %out = mlc.add %m, %f {target_device = #mlc.device<auto>} : tensor<1x2048xf16>
    return %out : tensor<1x2048xf16>
  }

  // Embedding + LM head — the two ops that bracket the layer stack at the
  // model level. Kept in the same fixture so the round-trip exercises every
  // op in the dialect.
  func.func @tinyllama_io(
      %ids:   tensor<1xi32>,
      %table: tensor<32000x2048xi8>,
      %lm_w:  tensor<2048x32000xi8>
  ) -> tensor<1x32000xf16> {
    %h = mlc.embedding %ids, %table
        {target_device = #mlc.device<auto>, quant_format = "q4_0"}
        : (tensor<1xi32>, tensor<32000x2048xi8>) -> tensor<1x2048xf16>
    %logits = mlc.lm_head %h, %lm_w
        {target_device = #mlc.device<auto>, quant_format = "q4_0"}
        : (tensor<1x2048xf16>, tensor<2048x32000xi8>) -> tensor<1x32000xf16>
    return %logits : tensor<1x32000xf16>
  }
}
