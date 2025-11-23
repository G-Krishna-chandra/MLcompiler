#include <gtest/gtest.h>
#include "ir/ir.hpp"
#include <sstream>

using namespace mlc::ir;

TEST(IRTest, DataTypeEnum) {
    // Test that DataType enum values exist
    DataType f32 = DataType::F32;
    DataType f16 = DataType::F16;
    DataType bf16 = DataType::BF16;
    DataType i8 = DataType::I8;
    DataType i4 = DataType::I4;
    
    EXPECT_NE(f32, f16);
    EXPECT_NE(i8, i4);
    EXPECT_NE(bf16, i4);
}

TEST(IRTest, OpKindEnum) {
    // Test that OpKind enum values exist
    OpKind matmul = OpKind::MatMul;
    OpKind add = OpKind::Add;
    OpKind linear = OpKind::Linear;
    
    EXPECT_NE(matmul, add);
    EXPECT_NE(add, linear);
}

TEST(IRTest, TensorCreation) {
    Tensor tensor("input", {1, 784}, DataType::F32, 0);
    
    EXPECT_EQ(tensor.name, "input");
    EXPECT_EQ(tensor.shape.size(), 2);
    EXPECT_EQ(tensor.shape[0], 1);
    EXPECT_EQ(tensor.shape[1], 784);
    EXPECT_EQ(tensor.dtype, DataType::F32);
    EXPECT_EQ(tensor.byteOffset, 0);
}

TEST(IRTest, NodeCreation) {
    Node node(OpKind::MatMul, "matmul1");
    
    EXPECT_EQ(node.kind, OpKind::MatMul);
    EXPECT_EQ(node.name, "matmul1");
    EXPECT_TRUE(node.inputs.empty());
    EXPECT_TRUE(node.outputs.empty());
    EXPECT_TRUE(node.attributes.empty());
}

TEST(IRTest, GraphAddTensor) {
    Graph graph;
    
    Tensor* t1 = graph.addTensor("tensor1", {1, 10}, DataType::F32);
    Tensor* t2 = graph.addTensor("tensor2", {10, 20}, DataType::F16);
    
    EXPECT_NE(t1, nullptr);
    EXPECT_NE(t2, nullptr);
    EXPECT_EQ(t1->name, "tensor1");
    EXPECT_EQ(t2->name, "tensor2");
    
    const auto& tensors = graph.tensors();
    EXPECT_EQ(tensors.size(), 2);
    EXPECT_EQ(tensors[0], t1);
    EXPECT_EQ(tensors[1], t2);
}

TEST(IRTest, GraphAddNode) {
    Graph graph;
    
    Node* n1 = graph.addNode(OpKind::MatMul, "matmul1");
    Node* n2 = graph.addNode(OpKind::Add, "add1");
    
    EXPECT_NE(n1, nullptr);
    EXPECT_NE(n2, nullptr);
    EXPECT_EQ(n1->name, "matmul1");
    EXPECT_EQ(n2->name, "add1");
    EXPECT_EQ(n1->kind, OpKind::MatMul);
    EXPECT_EQ(n2->kind, OpKind::Add);
    
    const auto& nodes = graph.nodes();
    EXPECT_EQ(nodes.size(), 2);
    EXPECT_EQ(nodes[0], n1);
    EXPECT_EQ(nodes[1], n2);
}

TEST(IRTest, NodeInputOutputRelationships) {
    Graph graph;
    
    Tensor* output1 = graph.addTensor("output1", {1, 20}, DataType::F32);
    
    // Create nodes
    Node* add_node = graph.addNode(OpKind::Add, "add1");
    Node* matmul_node = graph.addNode(OpKind::MatMul, "matmul1");
    
    // Set up relationships
    add_node->inputs.push_back(nullptr); // Can have null inputs for constants
    add_node->outputs.push_back(output1);
    
    matmul_node->inputs.push_back(add_node); // MatMul takes Add's output
    matmul_node->outputs.push_back(output1);
    
    // Verify relationships
    EXPECT_EQ(add_node->outputs.size(), 1);
    EXPECT_EQ(add_node->outputs[0], output1);
    
    EXPECT_EQ(matmul_node->inputs.size(), 1);
    EXPECT_EQ(matmul_node->inputs[0], add_node);
    EXPECT_EQ(matmul_node->outputs.size(), 1);
    EXPECT_EQ(matmul_node->outputs[0], output1);
}

TEST(IRTest, NodeAttributes) {
    Graph graph;
    
    Node* node = graph.addNode(OpKind::Norm, "norm1");
    node->attributes["epsilon"] = 1e-5f;
    node->attributes["axis"] = -1.0f;
    
    EXPECT_EQ(node->attributes.size(), 2);
    EXPECT_FLOAT_EQ(node->attributes["epsilon"], 1e-5f);
    EXPECT_FLOAT_EQ(node->attributes["axis"], -1.0f);
}

TEST(IRTest, ComplexGraphStructure) {
    Graph graph;
    
    // Create a simple computation graph: input -> linear -> add -> output
    Tensor* input = graph.addTensor("input", {1, 784}, DataType::F32);
    Tensor* weight = graph.addTensor("weight", {784, 128}, DataType::F32);
    Tensor* bias = graph.addTensor("bias", {128}, DataType::F32);
    ASSERT_NE(input, nullptr);
    ASSERT_NE(weight, nullptr);
    ASSERT_NE(bias, nullptr);
    Tensor* output = graph.addTensor("output", {1, 128}, DataType::F32);
    
    Node* linear = graph.addNode(OpKind::Linear, "linear1");
    linear->inputs.push_back(nullptr); // input handled separately
    linear->outputs.push_back(output);
    linear->attributes["in_features"] = 784.0f;
    linear->attributes["out_features"] = 128.0f;
    
    Node* add = graph.addNode(OpKind::Add, "add1");
    add->inputs.push_back(linear);
    add->outputs.push_back(output);
    
    // Verify graph structure
    EXPECT_EQ(graph.tensors().size(), 4);
    EXPECT_EQ(graph.nodes().size(), 2);
    
    EXPECT_EQ(linear->kind, OpKind::Linear);
    EXPECT_EQ(add->kind, OpKind::Add);
    EXPECT_EQ(add->inputs[0], linear);
}

TEST(IRTest, GraphDumpOutput) {
    Graph graph;
    
    Tensor* t1 = graph.addTensor("input", {1, 10}, DataType::F32);
    Tensor* t2 = graph.addTensor("output", {1, 20}, DataType::F16);
    (void)t1;
    
    Node* n1 = graph.addNode(OpKind::MatMul, "matmul1");
    n1->outputs.push_back(t2);
    n1->attributes["transpose_a"] = 0.0f;
    
    std::string dump = graph.dumpGraph();
    
    // Verify dump contains expected information
    EXPECT_NE(dump.find("Graph Dump"), std::string::npos);
    EXPECT_NE(dump.find("Tensors"), std::string::npos);
    EXPECT_NE(dump.find("Nodes"), std::string::npos);
    EXPECT_NE(dump.find("input"), std::string::npos);
    EXPECT_NE(dump.find("output"), std::string::npos);
    EXPECT_NE(dump.find("matmul1"), std::string::npos);
    EXPECT_NE(dump.find("MatMul"), std::string::npos);
    EXPECT_NE(dump.find("F32"), std::string::npos);
    EXPECT_NE(dump.find("F16"), std::string::npos);
}

TEST(IRTest, GraphDumpWithRelationships) {
    Graph graph;
    
    Tensor* input = graph.addTensor("input", {1, 10}, DataType::F32);
    Tensor* output = graph.addTensor("output", {1, 10}, DataType::F32);
    (void)input;
    
    Node* add = graph.addNode(OpKind::Add, "add1");
    add->inputs.push_back(nullptr);
    add->outputs.push_back(output);
    
    Node* mul = graph.addNode(OpKind::Mul, "mul1");
    mul->inputs.push_back(add);
    mul->outputs.push_back(output);
    
    std::string dump = graph.dumpGraph();
    
    // Verify relationships are shown
    EXPECT_NE(dump.find("add1"), std::string::npos);
    EXPECT_NE(dump.find("mul1"), std::string::npos);
    // The dump should show input/output relationships
    EXPECT_NE(dump.find("outputs="), std::string::npos);
}

TEST(IRTest, DataTypeToString) {
    EXPECT_EQ(dataTypeToString(DataType::F32), "F32");
    EXPECT_EQ(dataTypeToString(DataType::F16), "F16");
    EXPECT_EQ(dataTypeToString(DataType::BF16), "BF16");
    EXPECT_EQ(dataTypeToString(DataType::I8), "I8");
    EXPECT_EQ(dataTypeToString(DataType::I4), "I4");
}

TEST(IRTest, OpKindToString) {
    EXPECT_EQ(opKindToString(OpKind::MatMul), "MatMul");
    EXPECT_EQ(opKindToString(OpKind::Add), "Add");
    EXPECT_EQ(opKindToString(OpKind::Mul), "Mul");
    EXPECT_EQ(opKindToString(OpKind::Linear), "Linear");
    EXPECT_EQ(opKindToString(OpKind::Norm), "Norm");
    EXPECT_EQ(opKindToString(OpKind::Softmax), "Softmax");
    EXPECT_EQ(opKindToString(OpKind::Reshape), "Reshape");
    EXPECT_EQ(opKindToString(OpKind::Transpose), "Transpose");
}

TEST(IRTest, GraphMemoryManagement) {
    {
        Graph graph;
        
        // Create many nodes and tensors
        for (int i = 0; i < 10; ++i) {
            graph.addTensor("tensor" + std::to_string(i), {1, 10}, DataType::F32);
            graph.addNode(OpKind::Add, "node" + std::to_string(i));
        }
        
        EXPECT_EQ(graph.tensors().size(), 10);
        EXPECT_EQ(graph.nodes().size(), 10);
    }
    // Graph destructor should clean up all memory
    // If there are leaks, valgrind or sanitizers would catch them
}

TEST(IRTest, TensorByteOffset) {
    Graph graph;
    
    Tensor* t1 = graph.addTensor("t1", {1, 10}, DataType::F32);
    t1->byteOffset = 0;
    
    Tensor* t2 = graph.addTensor("t2", {1, 20}, DataType::F32);
    t2->byteOffset = 40; // 10 * 4 bytes (F32)
    
    EXPECT_EQ(t1->byteOffset, 0);
    EXPECT_EQ(t2->byteOffset, 40);
}
