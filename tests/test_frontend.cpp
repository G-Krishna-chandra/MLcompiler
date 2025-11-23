#include <gtest/gtest.h>
#include "frontends/frontend.hpp"

TEST(FrontendTest, BasicInitialization) {
    mlc::frontend::Frontend frontend;
    EXPECT_NO_THROW(frontend.analyze());
}

TEST(FrontendTest, ParseTest) {
    mlc::frontend::Frontend frontend;
    EXPECT_NO_THROW(frontend.parse("test source"));
}






