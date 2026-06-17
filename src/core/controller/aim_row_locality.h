#ifndef AIM_COSIM_CONTROLLER_AIM_ROW_LOCALITY_H
#define AIM_COSIM_CONTROLLER_AIM_ROW_LOCALITY_H

namespace aim_cosim {

enum class AiMRowLocalityClass {
    None = 0,
    ReadyLike = 1,
    Miss = 2,
    Conflict = 3,
};

struct AiMRowLocalityCommandIds {
    int act = -1;
    int act4 = -1;
    int act16 = -1;
    int pre = -1;
    int pre4 = -1;
    int prea = -1;
};

inline AiMRowLocalityClass classify_row_locality(
    int final_command,
    int preq_command,
    const AiMRowLocalityCommandIds& ids) {
    if (final_command < 0 || preq_command < 0) {
        return AiMRowLocalityClass::None;
    }
    if (preq_command == final_command) {
        return AiMRowLocalityClass::ReadyLike;
    }
    if (preq_command == ids.act || preq_command == ids.act4 || preq_command == ids.act16) {
        return AiMRowLocalityClass::Miss;
    }
    if (preq_command == ids.pre || preq_command == ids.pre4 || preq_command == ids.prea) {
        return AiMRowLocalityClass::Conflict;
    }
    return AiMRowLocalityClass::None;
}

} // namespace aim_cosim

#endif // AIM_COSIM_CONTROLLER_AIM_ROW_LOCALITY_H
