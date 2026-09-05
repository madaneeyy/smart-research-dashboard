import {
  useEffect,
  useState,
} from "react";
import { ChatPage } from "./components/ChatPage";
import { AppShell } from "./components/appshell";
import { CreateWorkspacePage } from "./components/CreateWorkspacePage";
import { LandingPage } from "./components/LandingPage";
import { WorkspaceOverview } from "./components/WorkspaceOverview";
import { SourcesPage } from "./components/SourcesPage";
import { DiscoverResearch } from "./components/DiscoverResearch";
import {
  createWorkspace,
  deleteWorkspace,
  getWorkspace,
  type Workspace,
} from "./lib/api";

/* ============================================================
   Types
   ============================================================ */

type Page =
  | "landing"
  | "create-workspace"
  | "workspace";

type AppSection =
  | "overview"
  | "sources"
  | "discover"
  | "chat";

type CreateWorkspaceReturn =
  | "landing"
  | "workspace";

/* ============================================================
   Persistence keys
   ============================================================ */

const ACTIVE_WORKSPACE_KEY =
  "smart-research-active-workspace";

const DEMO_WORKSPACE_KEY =
  "smart-research-demo-workspace";

const ACTIVE_SECTION_KEY =
  "smart-research-active-section";

  /* ============================================================
   Helpers
   ============================================================ */



/* ============================================================
   App
   ============================================================ */

function App() {
  const [page, setPage] =
    useState<Page>("landing");

  const [activeSection, setActiveSection] =
    useState<AppSection>("overview");

  const [workspace, setWorkspace] =
    useState<Workspace | null>(null);

  const [
    pendingChatDocumentIds,
    setPendingChatDocumentIds,
  ] = useState<string[]>([]);

  const [
    pendingChatGithubSourceIds,
    setPendingChatGithubSourceIds,
  ] = useState<string[]>([]);

  const [
    isRestoringWorkspace,
    setIsRestoringWorkspace,
  ] = useState(true);

  const [
    isCreatingWorkspace,
    setIsCreatingWorkspace,
  ] = useState(false);

  const [
    isDeletingWorkspace,
    setIsDeletingWorkspace,
  ] = useState(false);

  const [
    workspaceError,
    setWorkspaceError,
  ] = useState<string | null>(null);

  const [
    createWorkspaceReturn,
    setCreateWorkspaceReturn,
  ] =
    useState<CreateWorkspaceReturn>(
      "landing",
    );

  /* ==========================================================
     INITIAL
     ========================================================== */

  useEffect(() => {
    setIsRestoringWorkspace(false);
  }, []);

  /* ==========================================================
     DEMO
     ========================================================== */

  const handleTryDemo = async () => {
    setWorkspaceError(null);
    setIsRestoringWorkspace(true);

    try {
      const savedDemoWorkspaceId =
        window.localStorage.getItem(DEMO_WORKSPACE_KEY);

      /*
       * If this browser already has a demo workspace,
       * restore it.
       */
      if (savedDemoWorkspaceId) {
        try {
          const existingWorkspace =
            await getWorkspace(savedDemoWorkspaceId);

          setWorkspace(existingWorkspace);

          window.localStorage.setItem(
            ACTIVE_WORKSPACE_KEY,
            existingWorkspace.id,
          );

          window.localStorage.setItem(
            ACTIVE_SECTION_KEY,
            "overview",
          );

          setActiveSection("overview");
          setPage("workspace");

          return;
        } catch {
          /*
           * The saved workspace no longer exists.
           * Remove the stale browser reference and
           * show the Create Workspace page again.
           */
          window.localStorage.removeItem(
            DEMO_WORKSPACE_KEY,
          );

          window.localStorage.removeItem(
            ACTIVE_WORKSPACE_KEY,
          );
        }
      }

      /*
       * First demo visit for this browser.
       * Let the user configure their workspace through
       * the existing CreateWorkspacePage.
       */
      setCreateWorkspaceReturn("landing");
      setPage("create-workspace");
    } catch (error) {
      setWorkspaceError(
        error instanceof Error
          ? error.message
          : "Unable to open the demo workspace.",
      );
    } finally {
      setIsRestoringWorkspace(false);
    }
  };

  const handleSignIn = () => {
    console.log("Sign in clicked");
  };

  /* ==========================================================
     CREATE WORKSPACE
     ========================================================== */

  const handleBack = () => {
    if (
      isCreatingWorkspace ||
      isDeletingWorkspace
    ) {
      return;
    }

    setWorkspaceError(null);

    setPage(
      createWorkspaceReturn,
    );
  };

  const handleCreateWorkspace = async (
    data: {
      name: string;
      description: string;
      startMode:
        | "blank"
        | "topic"
        | "file";
    },
  ) => {
    setWorkspaceError(null);
    setIsCreatingWorkspace(true);

    try {
      const created =
        await createWorkspace({
          name: data.name,
          description:
            data.description || null,
        });

      setWorkspace(created);

      window.localStorage.setItem(
        ACTIVE_WORKSPACE_KEY,
        created.id,
      );

      window.localStorage.setItem(
        DEMO_WORKSPACE_KEY,
        created.id,
      );

      window.localStorage.setItem(
        ACTIVE_SECTION_KEY,
        "overview",
      );

      setActiveSection(
        "overview",
      );

      setPage("workspace");
    } catch (error) {
      setWorkspaceError(
        error instanceof Error
          ? error.message
          : "Unable to create workspace.",
      );
    } finally {
      setIsCreatingWorkspace(false);
    }
  };

  /* ==========================================================
     NAVIGATION
     ========================================================== */

  const handleNavigate = (
    section: AppSection,
  ) => {
    setActiveSection(section);

    window.localStorage.setItem(
      ACTIVE_SECTION_KEY,
      section,
    );
  };

  const handleNewWorkspace = () => {
    /*
     * Demo restriction:
     * account creation will be wired later.
     *
     * For now AppShell shows the account gate.
     */
    console.log(
      "Multiple workspaces require an account.",
    );
  };

  /* ==========================================================
     DELETE WORKSPACE
     ========================================================== */

  const handleDeleteWorkspace =
    async () => {
      if (
        !workspace ||
        isDeletingWorkspace
      ) {
        return;
      }

      setWorkspaceError(null);
      setIsDeletingWorkspace(true);

      try {
        await deleteWorkspace(
          workspace.id,
        );

        window.localStorage.removeItem(
          ACTIVE_WORKSPACE_KEY,
        );
        window.localStorage.removeItem(
          DEMO_WORKSPACE_KEY,
        );
        window.localStorage.removeItem(
          ACTIVE_SECTION_KEY,
        );

        setWorkspace(null);
        setActiveSection(
          "overview",
        );
        setPage("landing");
      } catch (error) {
        setWorkspaceError(
          error instanceof Error
            ? error.message
            : "Unable to delete workspace.",
        );
      } finally {
        setIsDeletingWorkspace(false);
      }
    };

  /* ==========================================================
     RESTORE
     ========================================================== */

  if (isRestoringWorkspace) {
    return <LoadingScreen />;
  }

  /* ==========================================================
     LANDING
     ========================================================== */

  if (page === "landing") {
    return (
      <LandingPage
        onTryDemo={handleTryDemo}
        onSignIn={handleSignIn}
      />
    );
  }

  /* ==========================================================
     CREATE WORKSPACE
     ========================================================== */

  if (
    page ===
    "create-workspace"
  ) {
    return (
      <CreateWorkspacePage
        onBack={handleBack}
        onCreate={
          handleCreateWorkspace
        }
        isCreating={
          isCreatingWorkspace
        }
        error={workspaceError}
      />
    );
  }

  /* ==========================================================
     WORKSPACE
     ========================================================== */

  if (
    page === "workspace" &&
    workspace
  ) {
    return (
      <AppShell
        workspace={workspace}
        activeSection={
          activeSection
        }
        onNavigate={
          handleNavigate
        }
        onCreateWorkspace={
          handleNewWorkspace
        }
        onDeleteWorkspace={
          handleDeleteWorkspace
        }
        isDeletingWorkspace={
          isDeletingWorkspace
        }
        workspaceError={
          workspaceError
        }
      >
        {activeSection ===
          "overview" && (
          <WorkspaceOverview
            workspace={workspace}
            onNavigate={
              handleNavigate
            }
            onCreateWorkspace={
              handleNewWorkspace
            }
          />
        )}

        {activeSection ===
          "sources" && (
          <SourcesPage
            workspace={workspace}
            onAskAI={(selectedSourceIds) => {
              const documentIds: string[] = [];
              const githubSourceIds: string[] = [];

              for (const rawId of selectedSourceIds) {
                const value = String(rawId).trim();

                if (!value) continue;

                if (value.toLowerCase().startsWith("github:")) {
                  const githubUrl = value.slice("github:".length).trim();
                  if (githubUrl) githubSourceIds.push(githubUrl);
                } else {
                  documentIds.push(value);
                }
              }

              setPendingChatDocumentIds(
                Array.from(new Set(documentIds)),
              );

              setPendingChatGithubSourceIds(
                Array.from(new Set(githubSourceIds)),
              );

              handleNavigate("chat");
            }}
          />
        )}

        {activeSection ===
          "discover" && (
          <DiscoverResearch
            workspace={workspace}
            onNavigateSources={()=>handleNavigate("sources")}
          />
        )}

        {activeSection ===
          "chat" && (
          <ChatPage
            workspace={workspace}
            initialDocumentIds={
              pendingChatDocumentIds
            }
            initialGithubSourceIds={
              pendingChatGithubSourceIds
            }
            onInitialDocumentsConsumed={() => {
              setPendingChatDocumentIds([]);
            }}
            onInitialGithubSourcesConsumed={() => {
              setPendingChatGithubSourceIds([]);
            }}
            onBack={() =>
              handleNavigate("overview")
            }
          />
        )}
      </AppShell>
    );
  }

  return (
    <LandingPage
      onTryDemo={handleTryDemo}
      onSignIn={handleSignIn}
    />
  );
}

/* ============================================================
   Loading
   ============================================================ */

function LoadingScreen() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--paper)] text-[var(--ink)]">
      <div className="flex flex-col items-center">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--ink)]">
          <span className="h-2 w-2 animate-pulse rounded-full bg-[var(--paper)]" />
        </div>

        <p className="mt-4 font-[var(--font-display)] text-sm font-semibold">
          Restoring workspace
        </p>

        <p className="mt-1 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.12em] text-[var(--muted)]">
          Please wait
        </p>
      </div>
    </div>
  );
}

/* ============================================================
   Placeholder
   ============================================================ */


export default App;