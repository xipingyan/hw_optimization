#pragma once

#if 1
#define DEBUG_LOG std::cout
#else
#define DEBUG_LOG std::cout << __FUNCTION__ << ":" << __LINE__ << " log: "
#endif