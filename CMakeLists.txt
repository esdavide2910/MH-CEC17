cmake_minimum_required(VERSION 3.0)

PROJECT(cec17)

ADD_EXECUTABLE(test "test.cc")
ADD_EXECUTABLE(testrandom "testrandom.cc")
ADD_EXECUTABLE(testsolis "testsolis.cc")
ADD_EXECUTABLE(testSSA "testSSA.cc")

ADD_LIBRARY("cec17_test_func" SHARED "cec17_test_func.c" "cec17.c")

TARGET_LINK_LIBRARIES(test "cec17_test_func")
TARGET_LINK_LIBRARIES(testrandom "cec17_test_func")
TARGET_LINK_LIBRARIES(testsolis "cec17_test_func")
TARGET_LINK_LIBRARIES(testSSA "cec17_test_func")

