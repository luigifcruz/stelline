#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <stelline/stacker/block.hh>
#include <jetstream/registry.hh>

using namespace Jetstream;

TEST_CASE("Stacker block declares its module",
          "[stelline][stacker][block][registration]") {
    const auto registrations = Registry::ListAvailableBlocks("stacker");
    REQUIRE(registrations.size() == 1);

    const std::vector<Registry::BlockModuleRequirement> expected = {
        {"stacker"},
    };
    REQUIRE(registrations.front().moduleRequirements == expected);
}
